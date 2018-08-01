# HawkesProcess_LSTM
--[[
A self-exciting event is something where one occurence of an event is more likely 
to trigger repeated occurences of that same event. For example, an event like an earthquake is likely 
to trigger succesive earthquake (aftershocks). Events of this nature are known as a Hawkes Process

The functions in this repoistory train an LSTM neural network to learn the elasped time between event 
occurences, known as inter-arrival times, in a Hawkes Process using statiscially 
simulated data for training. 

The run_LSTM.lua script initiates the process
]]
