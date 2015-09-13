{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE MultiParamTypeClasses #-}

--import qualified Data.Matrix as DM
--import qualified Data.Vector as DV

import Foreign.Storable
import Numeric.LinearAlgebra.Data
import Numeric.LinearAlgebra.HMatrix
import Numeric.LinearAlgebra

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
-------

newtype Channel a = Channel { recv :: a }
  deriving Show
send :: a -> Channel a
send = Channel

instance Monad Channel where
  c >>= f = f $ recv c 

instance Applicative Channel where
  pure    = send
  (<*>) f = fmap (recv f) 

instance Functor Channel where
  fmap f (Channel s) = Channel $ f s


---------------------
-- Network construction.

-- Data type to hold both activation
-- function and it's derivative.
data ActivationSpec a =
  ActivationSpec { actF  :: a -> a   -- activation function
                 , actF' :: a -> a   -- it's derivative
                 , desc  :: String } -- description
                 
instance Show (ActivationSpec a) where
  show = desc
           
-- Example of ActivationSpec for sigmoid function
sigmoidActSpec =
  ActivationSpec { actF  = sigmoid
                 , actF' = sigmoid'
                 , desc  = "Sigmoid activation function" }

-- Type for layer. Each layer may have
-- it's own activation function.
-- Neurons are represented by matrix
-- which rows are inputs and cols
-- are neurons. So, element on i'th
-- row and j'th column is a weight of
-- j'th neuron for i'th input.      
data Layer a =
  Layer { layerWeights :: Matrix a
        , layerActSpec :: ActivationSpec a }

instance (Show a, Storable a, Element a) => Show (Layer a) where
  show lr = "Weights: " ++ show (layerWeights lr) ++
            "\nActivation: " ++ show (layerActSpec lr) 
     
-- I decided to not keep learning rate value
-- with network. So this type only holds
-- network's layers ordered from 1'th
-- to last (output) layer.   
newtype Network a =
  Network { networkLayers :: [Layer a] }
  
instance (Show a, Storable a, Element a) => Show (Network a) where
  show (Network ls) = "Neural Network\n" ++
    (intercalate "\n" $
      map (\ (i, lr) ->
      "Layer " ++ show i ++ ":\n"
        ++ (indent indSize $ show lr)) (zip [1..] ls))
    where
      indSize = 4  
  
-- Number of neurons in layer. Auxiliary
-- data which describes a layer structure.
newtype LayerArch =
  LayerArch   { laNeuronNumber :: Int }

-- Layer's parameter which describes a layer.
-- Having this parameters and architecture
-- of input which this layer should accept,
-- one can can construct actual Layer.
data LayerParams a =
  LayerParams { lpLayerArch :: LayerArch
              , lpDefWeight :: a
              , lpActSpec   :: ActivationSpec a }

-- Layer may be constructed by it's input layer architecture
-- and parameters
initLayer :: Storable a => LayerArch -> LayerParams a -> Layer a
initLayer inputArch (LayerParams thisArch defWeight actSpec)  =
  let weights = (><)
        (laNeuronNumber inputArch)
        (laNeuronNumber thisArch) $
        repeat defWeight
        in Layer weights actSpec
      
-- Constructs a network by given input layer architecture
-- and parameters for each layer
initNetwork :: Storable a => LayerArch -> [LayerParams a] -> Network a
initNetwork inputArch params
  | null params = error "Network must have at least one layer"
  | otherwise =
    Network $ map (uncurry initLayer) $
      (inputArch, head params) : (map lpLayerArch params) `zip` (tail params)


------------------------------------------------------------------------------
-- Network propagation

-- To get network's output, one should
-- propagate network's input throught
-- all levels. We'll describe it as
-- new data type.
data PropagatedLayer a =
    SensorLayer { pLayerOutput  :: Vector a }
  | PropedLayer { pLayerInput   :: Vector a
                , pLayerOutput  :: Vector a
                , pLayerActF's  :: Vector a
                , pLayerWeights :: Matrix a
                , pLayerActSpec :: ActivationSpec a }
                
instance (Show a, Storable a, Element a) => Show (PropagatedLayer a) where
  show (SensorLayer output) =
    "Sensor layer: " ++ show output
  show (PropedLayer inp out f's ws as) =
    "Input:  " ++ show inp ++ "\n" ++
    "Output: " ++ show out ++ "\n" ++
    "F's:    " ++ show f's ++ "\n" ++
    "Weights: " ++ show ws ++ "\n" ++
    "Activation: " ++ desc as

                
propagateLayer :: Numeric a => PropagatedLayer a -> Layer a -> PropagatedLayer a
propagateLayer pl lr =
  PropedLayer { pLayerInput   = plOs
              , pLayerOutput  = lrFs
              , pLayerActF's  = lrF's
              , pLayerWeights = lrWs
              , pLayerActSpec = lrActSpec }
              where
                
                lrWs = layerWeights lr
                plOs = pLayerOutput pl
                
                lrActSpec = layerActSpec lr
                lrActF    = actF  lrActSpec
                lrActF'   = actF' lrActSpec
                
                lrSumS = trans lrWs #> plOs
                lrFs   = mapVector lrActF  lrSumS
                lrF's  = mapVector lrActF' lrSumS
                
newtype PropagatedNetwork a =
  PropagatedNetwork { pNetworkLayers :: [PropagatedLayer a] }
  
instance (Show a, Storable a, Element a) => Show (PropagatedNetwork a) where
  show (PropagatedNetwork ls) = "Propagated Neural Network\n" ++
    (intercalate "\n" $
      map (\ (i, lr) ->
      "Layer " ++ show i ++ ":\n"
        ++ (indent indSize $ show lr)) (zip [1..] ls))
    where
      indSize = 4
  
propagateNetwork :: Numeric a => PropagatedLayer a -> Network a -> PropagatedNetwork a
propagateNetwork inputLayer = PropagatedNetwork .
  tail . scanl propagateLayer inputLayer . networkLayers

-----------------------------------------------------------------
-- Network backward propagation

data BackPropedLayer a =
  BackPropedLayer { bpLayerInput   :: Vector a
                  , bpLayerOutput  :: Vector a
                  , bpLayerDeltas  :: Vector a
                  , bpLayerGrad    :: Matrix a
                  , bpLayerWeights :: Matrix a
                  , bpLayerActSpec :: ActivationSpec a }
                  
backPropOutputLayer :: (Show a, Numeric a, Num a, Num (Vector a)) =>
  PropagatedLayer a -> Vector a -> BackPropedLayer a
backPropOutputLayer outputLayer desiredOutput =
  BackPropedLayer { bpLayerInput   = is
                  , bpLayerOutput  = os
                  , bpLayerDeltas  = ds
                  , bpLayerGrad    = gr
                  , bpLayerWeights = ws
                  , bpLayerActSpec = pLayerActSpec outputLayer }
                  where
                    is = pLayerInput outputLayer
                    os = pLayerOutput outputLayer
                    ws = pLayerWeights outputLayer
                    ds = (os - desiredOutput) * pLayerActF's outputLayer
                    gr = errorGrad (rows ws, cols ws) is ds

errorGrad (r, c) inputs deltas = let
          lIs = toList inputs
          lDs = toList deltas
          in (r><c) $ concat $ map (zipWith (*) lDs . replicate c) lIs 
          
          --concatMap (\ d -> map (*d) lIs) lDs
                      
backPropHiddenLayer :: (Show a, Numeric a, Num a, Num (Vector a)) =>
  PropagatedLayer a -> BackPropedLayer a -> BackPropedLayer a
backPropHiddenLayer thisLayer nextLayer =
  BackPropedLayer { bpLayerInput   = is
                  , bpLayerOutput  = os
                  , bpLayerDeltas  = ds
                  , bpLayerGrad    = gr
                  , bpLayerWeights = ws
                  , bpLayerActSpec = pLayerActSpec thisLayer }
                  where
                    ws = pLayerWeights thisLayer
                    is = pLayerInput thisLayer
                    os = pLayerOutput thisLayer
                    ds = (bpLayerWeights nextLayer #> bpLayerDeltas nextLayer) * pLayerActF's thisLayer
                    gr = errorGrad (rows ws, cols ws) is ds
                    
newtype BackPropedNetwork a =
  BackPropedNetwork { bpNetworkLayers :: [BackPropedLayer a] }
                    
backPropNetwork :: (Show a, Numeric a, Num a, Num (Vector a)) =>
  Vector a -> PropagatedNetwork a -> BackPropedNetwork a
backPropNetwork desiredOutput (PropagatedNetwork pLs) =
  let
    bpOutputLayer = backPropOutputLayer (last pLs) desiredOutput
    in BackPropedNetwork $ scanr backPropHiddenLayer bpOutputLayer $ init pLs
    
----------------------------------------------------------------------------------
-- Updating weights

updateLayer :: (Numeric a, Num a, Num (Vector a)) => a -> BackPropedLayer a -> Layer a
updateLayer rate bpLayer =
  let ws = bpLayerWeights bpLayer - rate `scale` bpLayerGrad bpLayer
    in Layer { layerWeights = ws
             , layerActSpec = bpLayerActSpec bpLayer }
             
updateNetwork :: (Numeric a, Num a, Num (Vector a)) => a -> BackPropedNetwork a -> Network a
updateNetwork rate = Network . map (updateLayer rate) . bpNetworkLayers 

-----------------------------------------------------------------------------------
-- Training

sensorLayer :: Vector a -> PropagatedLayer a
sensorLayer outputs = SensorLayer outputs

trainSample :: (Show a, Numeric a, Num a, Num (Vector a)) =>
  a -> (Vector a, Vector a) -> Network a -> Network a
trainSample rate (testInp, testOut) net =
  let sLr   = sensorLayer testInp
      pNet  = propagateNetwork sLr net
      bpNet = backPropNetwork testOut pNet
      in updateNetwork rate bpNet

trainSamples :: (Show a, Numeric a, Num a, Num (Vector a)) =>
  a -> [(Vector a, Vector a)] -> Network a -> Network a
trainSamples rate samples net = last $ scanl (flip (trainSample rate)) net samples 


------------------------------------------------------------
-- Tests, examples ...

runNet inp = pLayerOutput . last . pNetworkLayers . propagateNetwork (sensorLayer inp)

type TrainSample = (Vector Double, Vector Double)
 
xorTest1 :: (Vector Double, Vector Double)
xorTest1 = (2 |> [1, 1], 1 |> [0])

xorTest2 :: (Vector Double, Vector Double)
xorTest2 = (2 |> [1, 0], 1 |> [1])

xorTest3 :: (Vector Double, Vector Double)
xorTest3 = (2 |> [0, 1], 1 |> [1])

xorTest4 :: (Vector Double, Vector Double)
xorTest4 = (2 |> [0, 0], 1 |> [0])

xorSamples = [xorTest1, xorTest2, xorTest3, xorTest4]

xorNet = initNetwork (LayerArch 2) $
  [LayerParams (LayerArch 6) 1 sigmoidActSpec
  ,LayerParams (LayerArch 8) 1 sigmoidActSpec
  ,LayerParams (LayerArch 1) 1 sigmoidActSpec]

------------------------------------------------------------
-- Network sample
sampleNetwork :: Network Double      
sampleNetwork =
  initNetwork (LayerArch 3) $
    [LayerParams (LayerArch 5) 1 sigmoidActSpec
    ,LayerParams (LayerArch 4) 2 sigmoidActSpec
    ,LayerParams (LayerArch 3) 3 sigmoidActSpec]
   
getLayer net num = networkLayers net !! (num - 1) 
 

sampleMatrix :: (Num a, Storable a) => Matrix a                
sampleMatrix = (><) 4 3 $ repeat 1

sampleVector :: (Num a, Storable a) => Vector a
sampleVector = (|>) 4 $ repeat 1

 
sensorLayerSample :: (Num a, Storable a) => PropagatedLayer a
sensorLayerSample = SensorLayer $ 3 |> repeat 1
