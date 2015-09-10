
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


-- Neuron holds only his weights and threshold value
data Neuron w =
  Neuron { weights   :: [w]
         , threshold :: w }

-- Activation must be smooth function from one argument
type Activation a = a -> a
-- Activator is a pair from such function and a channels to receive inputs
type Activator a = (Activation a, [Channel a])

-- Forces neuron to send response on given activator
activate :: Num a => Activator a -> Neuron a -> Channel a
activate (f, cs) n =
  sequence cs >>= return . f . subs (threshold n) . dotProd (weights n)
  
------------------------------------------
-- Netwok --
{-
  I wanted to experiment with Vectors and Matrixes
  and see how these immutable structurse may influe
  on overal project here.
-}


-- Id's are for implementation. They won't
-- be used by user (i think).
newtype LayerId   = LayerId Int
newtype NeuronId  = NeuronId Int
newtype ChannelId = ChannelId Int

-- Node of network:
-- input channels, output channel and processor
data Node w =
  Node { input  :: [ChannelId]
       , output :: ChannelId
       , proc   :: Neuron w }

-- Layer just stores neurons
newtype Layer w = Layer { units :: DV.Vector (Node w) }

-- Network consists from layers which comes in order
-- and channels matrix. Neurons reads out and writes
-- in there. Here, rows refers to LayerId's. Cols refers
-- to ChannelId's.
data Network a =
  Network { layers   :: DV.Vector (Layer a)
          , channels :: DM.Matrix (Channel a) }
          
-- Constant for id of input layer.
inputLayerId :: LayerId
inputLayerId = LayerId (-1)         
 

