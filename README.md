# backprop-nn
Neural network for backward propagation algorithm.

This is a colleague work. Inspired by this solution:
https://github.com/mhwombat/backprop-example

I used approach of typing Network's stages here.

Everything is in main.hs now. I'll fix it later maybe.

Working example:
  
  type in GHCi
  
    let net = trainSamplesN 1000 0.8 digitSamples digitNet
    
  to train network to recognize digits made of plates like
  
    one = [ [0, 1, 0]
          , [1, 1, 0]
          , [0, 1, 0]
          , [0, 1, 0]
          , [0, 1, 0] ]
          
  There are four plates available (see main.hs): one1, one2, two, three.
  
  To get result, run:
  
    runNetwork (15 |> concat three) $ snd net
 
  This will give you the ouput
  
    [1.1323782478389121e-2,1.5553882684473243e-2,0.9814364715741604]
    
  which tells that digit 3 was recognized.
  
  This example is pretty odd. I'll fix it later.
