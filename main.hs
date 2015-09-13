{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE MultiParamTypeClasses #-}

--import qualified Data.Matrix as DM
--import qualified Data.Vector as DV

import Foreign.Storable
import Numeric.LinearAlgebra.Data
import Numeric.LinearAlgebra.HMatrix
import Numeric.LinearAlgebra hiding ((<>))

import Data.List

import Control.Monad.State
import Control.Monad.Trans.Maybe
import Control.Monad
import Data.Maybe

import Debug.Trace

-------
-- Auxiliaies ...

dotProd :: Num a => [a] -> [a] -> a
dotProd x = sum . zipWith (*) x

sigmoid :: Floating a => a -> a         
sigmoid x = 1 / ( 1 + exp (-x) )

sigmoid' :: Floating a => a -> a
sigmoid' x = sigmoid x * (1 - sigmoid x)

subs :: Num a => a -> a -> a
subs x y = y - x

pair = (,)

indent :: Int -> String -> String
indent spaceSize =
  let space = replicate spaceSize ' '
    in init . unlines . map (space++) . lines
    
showLayers :: Show a => [a] -> String
showLayers = intercalate "\n" .
               map (\ (i, x) ->
                 "Layer " ++ show i ++ ":\n" ++ indent indSz (show x)
                 ) . zip [1..] 
  where
    indSz = 4
    
ntimes n f = foldr (.) id (replicate n f)
-------

{-
  There are a three stages of neural network working:
  
  1) (not formal) to construct network
  2) to propagate input forward
  3) to propagate error backward
-}

-----------------------------------------------------
-- Stage 1. Construction.

-- Neuron's activation
data ActivationSpec a =
  ActivationSpec { asFunc  :: a -> a   -- activation function
                 , asFunc' :: a -> a   -- it's derivative
                 , asDesc  :: String } -- description

instance Show (ActivationSpec a) where
  show = asDesc

-- Network Layer
data Layer a =
  Layer { lWs :: Matrix a           -- layer weights
        , lAS :: ActivationSpec a } -- layer activation (one activation for each neuron per layer)

instance (Show a, Element a) => Show (Layer a) where
  show (Layer ws as) =
    "Weights: " ++ show ws ++ "\nActivation: " ++ show as
    
-- Network
newtype Network a =
  Network { nLs :: [Layer a] }
  
instance (Show a, Element a) => Show (Network a) where
  show = (++) "Network\n" . showLayers . nLs
  
-- Assembling
newtype LayerSize = LayerSize { lsVal :: Int } -- number of neurons

data LayerParams a =
  LayerParams { lpSz :: LayerSize
              , lpW  :: a
              , lpAS :: ActivationSpec a }
              
initLayer :: (Storable a) => LayerSize -> LayerParams a -> Layer a
initLayer inputSize params =
  let
    r  = lsVal inputSize
    c  = lsVal $ lpSz params
    ws = (r><c) $ repeat $ lpW params
    as = lpAS params
    in Layer ws as
    
initNetwork :: (Storable a) => LayerSize -> [LayerParams a] -> Network a
initNetwork inputSize params =
  Network $ map (uncurry initLayer) $
    (inputSize, head params) : (map lpSz params) `zip` tail params
    
    
-----------------------------------------------------
-- Stage 2. Forward Propagation.

data PropedLayer a =
    SensorLayer { plOutput :: Vector a }
    
  | PropedLayer { plInput  :: Vector a
                , plOutput :: Vector a
                , plF's    :: Vector a
                , plWs     :: Matrix a
                , plAS     :: ActivationSpec a }
                

instance (Show a, Element a) => Show (PropedLayer a) where
  show (SensorLayer out) =
    "SensorLayer: " ++ show out
  show (PropedLayer inp out f's ws as) =
    "Input:  "     ++ show inp ++ "\n" ++
    "Output: "     ++ show out ++ "\n" ++
    "F's:    "     ++ show f's ++ "\n" ++
    "Weights: "    ++ show ws  ++ "\n" ++
    "Activation: " ++ show as
   
newtype PropedNetwork a =
  PropedNetwork { pnLs :: [PropedLayer a] }   

instance (Show a, Element a) => Show (PropedNetwork a) where
  show = (++) "Propagated Network\n" . showLayers . pnLs

propLayer :: Numeric a => PropedLayer a -> Layer a -> PropedLayer a
propLayer pLr lr =
  let lrInp  = plOutput pLr
      lrWs   = lWs lr
      lrAS   = lAS lr
      lrSums = trans lrWs #> lrInp
      lrOut  = mapVector (asFunc lrAS) lrSums
      lrF's  = mapVector (asFunc' lrAS) lrSums 
      in PropedLayer { plInput  = lrInp
                     , plOutput = lrOut
                     , plF's    = lrF's
                     , plWs     = lrWs
                     , plAS     = lrAS }

propNetwork :: Numeric a => PropedLayer a -> Network a -> PropedNetwork a
propNetwork sensorLr =
  PropedNetwork . tail . scanl propLayer sensorLr . nLs 

-----------------------------------------------------
-- Stage 3. Backward Propagation.
data BackPropedLayer a =
  BackPropedLayer { bplDeltas :: Vector a
                  , bplGrad   :: Matrix a
                  , bplWs     :: Matrix a
                  , bplAS     :: ActivationSpec a }
                  
instance (Show a, Element a) => Show (BackPropedLayer a) where
  show (BackPropedLayer ds grad ws as) =
    "Error Deltas: "   ++ show ds   ++ "\n" ++
    "Error Gradient: " ++ show grad ++ "\n" ++
    "Weights:"         ++ show ws   ++ "\n" ++
    "Activation: "     ++ show as

newtype BackPropedNetwork a =
  BackPropedNetwork { bpnLs :: [BackPropedLayer a] }

instance (Show a, Element a) => Show (BackPropedNetwork a) where
  show = (++) "Backpropagated Network\n" . showLayers . bpnLs

backPropOutputLayer :: (Storable a, Num a, Num (Vector a)) => PropedLayer a -> Vector a -> BackPropedLayer a
backPropOutputLayer outputLayer desiredOutput =
  let
    lrWs   = plWs outputLayer
    lrInp  = plInput outputLayer
    lrOut  = plOutput outputLayer
    lrF's  = plF's outputLayer
    lrDs   = (lrOut - desiredOutput) * lrF's
    lrGrad = errorGrad (rows lrWs, cols lrWs) lrInp lrDs
    in BackPropedLayer { bplDeltas = lrDs
                       , bplGrad   = lrGrad
                       , bplWs     = lrWs
                       , bplAS     = plAS outputLayer }

-- works as intended    
errorGrad :: (Num a, Storable a) => (Int, Int) -> Vector a -> Vector a -> Matrix a
errorGrad (r, c) input deltas =
  let
    lInp = toList input
    lDs  = toList deltas
    in (r><c) $ concatMap (zipWith (*) lDs . repeat) lInp
    
backPropHiddenLayer :: (Numeric a, Num a, Num (Vector a)) =>
  PropedLayer a -> BackPropedLayer a -> BackPropedLayer a
backPropHiddenLayer thisLayer nextLayer =
  let nextLrWs   = bplWs nextLayer
      nextLrDs   = bplDeltas nextLayer
      thisLrF's  = plF's thisLayer
      thisLrDs   = (nextLrWs #> nextLrDs) * thisLrF's
      thisLrWs   = plWs thisLayer
      thisLrInp  = plInput thisLayer
      thisLrGrad = errorGrad (rows thisLrWs, cols thisLrWs) thisLrInp thisLrDs
      in BackPropedLayer { bplDeltas = thisLrDs
                         , bplGrad   = thisLrGrad
                         , bplWs     = thisLrWs
                         , bplAS     = plAS thisLayer }
                         
backPropNetwork :: (Numeric a, Num a, Num (Vector a)) =>
  Vector a -> PropedNetwork a -> BackPropedNetwork a
backPropNetwork desiredOutput propedNet =
  let
    netLs          = pnLs propedNet
    lastBackProped = backPropOutputLayer (last netLs) desiredOutput
    in BackPropedNetwork $ scanr backPropHiddenLayer lastBackProped $ init netLs 

-----------------------------------------------------
-- Stage 4. Updating.

-- `Numeric` to infer `Container Vector a`
updateLayer :: (Numeric a, Num a, Num (Vector a)) =>
  a -> BackPropedLayer a -> Layer a
updateLayer rate bpLr =
  let
    bpLrWs = bplWs bpLr
    bpLrG  = bplGrad bpLr
    lrWs   = bpLrWs - rate `scale` bpLrG
    lrAS   = bplAS bpLr
    in Layer lrWs lrAS

updateNetwork :: (Numeric a, Num a, Num (Vector a)) =>
  a -> BackPropedNetwork a -> Network a
updateNetwork rate = Network . map (updateLayer rate) . bpnLs

-----------------------------------------------------
-- Stage 5. Utiles.

runNetwork :: (Numeric a) => Vector a -> Network a -> Vector a
runNetwork inp = plOutput . last . pnLs . propNetwork (SensorLayer inp) 

-----------------------------------------------------
-- Stage 5. Traning.

type Sample a = (Vector a, Vector a)

trainSample :: (Numeric a, Num a, Num (Vector a)) =>
  a -> Sample a -> Network a -> (Vector a, Network a)
trainSample rate (inp, out) net =
  let
    propedNet = propNetwork (SensorLayer inp) net
    propedOut = plOutput $ last $ pnLs propedNet
    backprNet = backPropNetwork out propedNet
    in (out - propedOut, updateNetwork rate backprNet)

trainSampleN :: (Numeric a, Num a, Num (Vector a)) =>
  Int -> a -> Sample a -> Network a -> (Vector a, Network a)
trainSampleN n rate sample = ntimes n goTrain . (,) 1
  where
    goTrain (_, net) = trainSample rate sample net
    
trainSamples :: (Numeric a, Num a, Num (Vector a)) =>
  a -> [Sample a] -> Network a -> (Vector a, Network a)
trainSamples rate samples net = 
  foldl (\ (_, net) s -> trainSample rate s net) (1, net) samples  

trainSamplesN :: (Numeric a, Num a, Num (Vector a)) =>
  Int -> a -> [Sample a] -> Network a -> (Vector a, Network a)
trainSamplesN n rate samples = ntimes n goTrain . (,) 1
  where
    goTrain (_, net) = trainSamples rate samples net

-- Samples

sigmoidAS :: Floating a => ActivationSpec a
sigmoidAS =
  ActivationSpec { asFunc  = sigmoid
                 , asFunc' = sigmoid'
                 , asDesc    = "Sigmoid" }
                 
sampleNetwork :: Network Double      
sampleNetwork =
  initNetwork (LayerSize 2) $
    [LayerParams (LayerSize 3) 1 sigmoidAS
    ,LayerParams (LayerSize 2) 2 sigmoidAS
    ,LayerParams (LayerSize 1) 3 sigmoidAS]
    
sampleLayer :: Layer Double
sampleLayer = initLayer (LayerSize 3) $
                LayerParams (LayerSize 5) 1 sigmoidAS
    
v1 :: Vector Double
v1 = 3 |> [1,2,-1]

v2 :: Vector Double
v2 = 2 |> [0.5, 0.3]

---------------
-- Digits

type Digit = [[Int]]

one1 = [ [0, 1, 0]
       , [1, 1, 0]
       , [0, 1, 0]
       , [0, 1, 0]
       , [0, 1, 0] ]
      
one2 = [ [0, 0, 1]
       , [0, 1, 1]
       , [0, 0, 1]
       , [0, 0, 1]
       , [0, 0, 1] ]
      
two = [ [0, 1, 0]
      , [1, 0, 1]
      , [0, 1, 0]
      , [1, 0, 0]
      , [1, 1, 1] ]
      
three = [ [1, 1, 0]
        , [0, 0, 1]
        , [0, 1, 0]
        , [0, 0, 1]
        , [1, 1, 0] ]

digitSamples :: [Sample Double]
digitSamples =
  [(15 |> concat one1,  3 |> [1, 0, 0])
  ,(15 |> concat one2,  3 |> [1, 0, 0])
  ,(15 |> concat two,   3 |> [0, 1, 0])
  ,(15 |> concat three, 3 |> [0, 0, 1])]

digitNet :: Network Double
digitNet = initNetwork (LayerSize 15) $
  [LayerParams (LayerSize 3) 1 sigmoidAS]
  
--let net = trainSamplesN 1000 0.8 digitSamples digitNet

---------------
-- XOR


xorSample1 :: Sample Double
xorSample1 = (2 |> [1, 1], 1 |> [0])

xorSample2 :: Sample Double
xorSample2 = (2 |> [1, 0], 1 |> [1])

xorSample3 :: Sample Double
xorSample3 = (2 |> [0, 1], 1 |> [1])

xorSample4 :: Sample Double
xorSample4 = (2 |> [0, 0], 1 |> [0])

xorSamples = [xorSample1, xorSample2, xorSample3]--, xorSample4]

xorNet :: Network Double
xorNet = initNetwork (LayerSize 2) $
  [LayerParams (LayerSize 3) 1 sigmoidAS
  ,LayerParams (LayerSize 1) 1 sigmoidAS]

{-
xorSample1 :: Sample Double
xorSample1 = (2 |> [1, 1], 4 |> [1, 0, 0, 0])

xorSample2 :: Sample Double
xorSample2 = (2 |> [1, 0], 4 |> [0, 1, 0, 0])

xorSample3 :: Sample Double
xorSample3 = (2 |> [0, 1], 4 |> [0, 0, 1, 0])

xorSample4 :: Sample Double
xorSample4 = (2 |> [0, 0], 4 |> [0, 0, 0, 1])

xorSamples = [xorSample1, xorSample2, xorSample3, xorSample4]

xorNet :: Network Double
xorNet = initNetwork (LayerSize 2) $
  [LayerParams (LayerSize 4) 1 sigmoidAS]-}
