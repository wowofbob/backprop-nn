
import qualified Data.Vector as DV

-------
-- Auxiliaies ...

dotProd :: Num a => DV.Vector a -> DV.Vector a -> a
dotProd x = DV.sum . DV.zipWith (*) x

sigmoid :: Floating a => a -> a -> a         
sigmoid p x = 1 / ( 1 + exp ((-x)/p) )

-------

data Neuron =
  Neuron { neuronInputs    :: DV.Vector Double
         , neuronThreshold :: Double }
        
type Activation = Double -> Double -> Double
        
sumInput :: Neuron -> DV.Vector Double -> Double
sumInput n is = dotProd (neuronInputs n) is - (neuronThreshold n)

activate :: Double -> Activation -> Neuron -> DV.Vector Double -> Double
activate response f n = f response . sumInput n

-- Layer is parametrized. It's better them parametrize neurons.
data Layer =
  Layer { layerNeurons    :: DV.Vector Neuron
        , layerResponse   :: Double
        , layerActivation :: Activation } 



{-
         , neuronResponse   :: Double
         , neuronActivation :: Double -> Double }-}
