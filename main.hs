
import qualified Data.Vector as DV

type Input = Float
type Weight = Float

-- Every neuron is parametrized. For first time.
data Neuron =
  Neuron { neuronInputs     :: DV.Vector Double
         , neuronThreshold  :: Double
         , neuronResponse   :: Double
         , neuronActivation :: Double -> Double }
