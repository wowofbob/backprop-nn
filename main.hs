
import qualified Data.Matrix as DM
import qualified Data.Vector as DV

-------
-- Auxiliaies ...

vdotProd :: Num a => DV.Vector a -> DV.Vector a -> a
vdotProd x = DV.sum . DV.zipWith (*) x

dotProd :: Num a => [a] -> [a] -> a
dotProd x = sum . zipWith (*) x

sigmoid :: Floating a => a -> a -> a         
sigmoid p x = 1 / ( 1 + exp ((-x)/p) )

subs :: Num a => a -> a -> a
subs x y = y - x

pair = (,)
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
-- Network   

newtype Arch = Arch { archToList :: [Int] }

nullArch (Arch xs) = null xs

data Neuron w =
  Neuron { weights   :: [w]
         , threshold :: w }
          deriving Show
         
type Activation  a = a -> a -- activation
type Activation' a = a -> a -- it's derivation
         
newtype Layer w = Layer { neurons :: [Neuron w] }
  deriving Show

-- back propagation network
data BPNetwork a =
  BPNetwork { arch      :: Arch                -- network architecture
            , act       :: Activation  a       -- neuron activation
            , act'      :: Activation' a       -- neuron activation derivative
            , layers    :: [Layer a]           -- layers of neurons
            , lrate     :: a }                 -- learning rate
            
            
type Activator a = (Activation a, [a])
            
activate :: Num a => Activator a -> Neuron a -> a
activate (f, is) =
  f . getSum is
  
getSum :: Num a => [a] -> Neuron a -> a
getSum is n = weights n `dotProd` is - threshold n
            
            
actLayer :: Num a => Activator a -> Layer a -> [a]
actLayer a l = neurons l >>= return . activate a
            
runBPNetwork :: Num a => BPNetwork a -> [a] -> [a]
runBPNetwork net inp =
  foldl (\ is lr -> actLayer (actF, is) lr) inp (layers net)
    where
      actF = act net

genBPNetwork :: Num a => Arch -> Activation a -> Activation' a -> a -> BPNetwork a
genBPNetwork netArch actF actF' lr
  | nullArch netArch = error "Given Architecture is undefined"
  | otherwise =
    BPNetwork netArch actF actF' genLayers lr
    where
      
      inpLayerLen = head $ archToList netArch
      
      -- samples 
      sampleWeight = 1
      sampleThresh = 1
      
      sampleNeuron wLen =
        Neuron (replicate wLen sampleWeight) sampleThresh  
      
      sampleLayer predLen thisLen =
        Layer $ replicate thisLen $ sampleNeuron predLen
      
      -- layers generator
      genLayers =
        reverse $ snd $ foldl (\ (predLen, ls) thisLen ->
                (thisLen, sampleLayer predLen thisLen : ls)) (inpLayerLen, []) $
                  tail $ archToList netArch
                  
                  
-- training
--train :: Num a => BPNetwork a -> ([a], [a]) -> BPNetwork a
train net (inputs, correctOutputs) = go inputs (layers net)
  where
  
    actF' = act' net
    actF  = act  net
    nrate = negate $ lrate net
    
    
    go inputs [lastLayer] =
      let (sums, outs) = unzip $ map (\ n -> let
                                        s = getSum inputs n
                                        o = actF s
                                        in (s, o)) $ neurons lastLayer
          -- верно :
          deltas = zipWith (*) (map actF' sums) (zipWith (-) outs correctOutputs)
          
          dWs    = map (\ d -> map (*(nrate * d)) inputs) deltas
      in dWs
      
    updateWeights dWs = undefined
