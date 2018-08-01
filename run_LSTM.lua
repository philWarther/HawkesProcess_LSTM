-- run_LSTM.lua

require 'pause'
require 'train_hawkes_LSTM'

local dim = {[1] = 10, [0] = 1}
local dt0 = 1.25
local step_reduction = 0.75


train_hawkes_LSTM(dim, dt0, step_reduction)
