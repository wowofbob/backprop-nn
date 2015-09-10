
import qualified Data.Vector as DV

type Input = Float
type Weight = Float

data Conn =
  { connWeight :: Weight
  , connSignal :: Input }

data Neuron =
  Neuron { neuronInput :: DV.Vector Conn }
