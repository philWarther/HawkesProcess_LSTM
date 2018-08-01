function train_hawkes_LSTM(dim, dt0, step_reduction)


--[[
A self-exciting event is something where one occurence of an event is more likely 
to trigger repeated occurences of that same event. For example, an event like an earthquake is likely 
to trigger succesive earthquake (aftershocks). Events of this nature are known as a Hawkes Process

This function trains an LSTM neural network to learn elasped time between event 
occurences, known as inter-arrival times, in a Hawkes Process using statiscially 
simulated data for training.

Validation for the network is computed by comparing the error of the predicted inter-arrival times
to the average inter-arrival. The relative error (or error relative to the average) can be seen 
as a percentage improvement over using the average inter-arrival time to predict event occurences

INPUTS

dim -- 				A table denoting the dimensions of the single-layer network,
					the zero index of this table (dim[0]) indicates the dimension of
					the data input

dt0 --				The initial learning rate for the gradient descent algorithm

step_reduction -- 	When learning shows no improvment over several epochs, the learning rate is 
					mulitplied by this number to improve learning
					note: The stopping criteria of the training process is triggered once the 
					learning rate is reduced in size to a certain threshold
]]

require 'nn'
require 'nngraph'
require 'xlua'
require 'HawkesNdim'
require 'getmodule'
require 'pause'


-- Parameters for sequence generation

no_of_processes	= 1
epsilon 		= 10^(-8)
alpha 			= torch.Tensor(dim[0]):fill(1.1)--uniform(.5,5)
beta 			= torch.ones(dim[0],dim[0]):fill(1.5)--uniform(.5,8)
delta 			= torch.ones(dim[0]):fill(1.0)--uniform(.1,5)
lambda0 		= torch.ones(dim[0]):fill(2.0)--uniform(alpha[1] + epsilon, alpha[1] + 5)

-- Initializing Tables
model 				= {}
model_table 	 	= {}
--hdim				= {}


event_count			= 100				-- Length of the data sequence
criterion 			= nn.MSECriterion() -- Mean Squared Error Criterion Function
criterion.sizeAverage = false

local function LSTMcell()
	
	local x 	= nn.Identity()() -- eventually x
	local h 	= nn.Identity()() -- eventually state[2*i - 1] for ith layer
	local c 	= nn.Identity()() -- eventually state[2*i] for ith layer

	local i = 1
	
	local input_in 	= nn.Add(dim[i])(nn.CAddTable()({nn.Linear(dim[i-1],dim[i],false)(x),nn.Linear(dim[i],dim[i],false)(h)}))
	local forget_in = nn.Add(dim[i])(nn.CAddTable()({nn.Linear(dim[i-1],dim[i],false)(x),nn.Linear(dim[i],dim[i],false)(h)}))
	local output_in = nn.Add(dim[i])(nn.CAddTable()({nn.Linear(dim[i-1],dim[i],false)(x),nn.Linear(dim[i],dim[i],false)(h)}))
	local tanh_in 	= nn.Add(dim[i])(nn.CAddTable()({nn.Linear(dim[i-1],dim[i],false)(x),nn.Linear(dim[i],dim[i],false)(h)}))
	
	local input_gate 	= nn.Sigmoid()(input_in)
	local forget_gate 	= nn.Sigmoid()(forget_in)
	local output_gate 	= nn.Sigmoid()(output_in)
	local tanh_gate		= nn.Tanh()(tanh_in)
	
	cell_new 		= nn.CAddTable()({nn.CMulTable()({forget_gate,c}),nn.CMulTable()({input_gate,tanh_gate})})
	h_new			= nn.CMulTable()({output_gate, cell_new})
	x_new			= nn.Linear(dim[i],dim[0])(h_new)

	local model 	= nn.gModule({x,h,c},{x_new,h_new,cell_new})

	return model
end



local function replicate_module()
	mx,mdx = model:parameters()
	for i = 1,#mx do
		mx[i]:uniform(-0.07, 0.07)
	end

	for t = 1,event_count do
		model_table[t]=model:clone()
		px,pdx = model_table[t]:parameters()

		for i = 1,#px do
			px[i]:set(mx[i])
			pdx[i]:set(mdx[i])
			
		end
	end

	collectgarbage()
end

local function forward_pass()
	local model_error 	= 0
	local average_error = 0
	
	local y_out 	= model_table[1]:forward({torch.Tensor(dim[0]):fill(inputs[1]), state_in[1], state_in[2]})
	local crit_out 	= criterion:forward(y_out[1],torch.Tensor(1):fill(targets[1]))
	local avg_out 	= criterion:forward(torch.Tensor(1):fill(average),torch.Tensor(1):fill(targets[1]))

	model_error 	= model_error + crit_out
	average_error 	= average_error + avg_out

	for t= 2, event_count do
		
		for l = 2,3 do
			state[l-1]:copy(model_table[t-1].output[l])
		end
		local y_out 	= model_table[t]:forward({torch.Tensor(dim[0]):fill(inputs[t]),state[1],state[2]})
		local crit_out	= criterion:forward(y_out[1],torch.Tensor(1):fill(targets[t]))
		local avg_out 	= criterion:forward(torch.Tensor(1):fill(average),torch.Tensor(1):fill(targets[t]))

		model_error 	= model_error + crit_out
		average_error 	= average_error + avg_out

	end
	for l = 1,3 do
		state_out[l-1]:copy(model_table[event_count].output[l])
	end

	return model_error,average_error 
end

local function backward_pass()
	local dG = {}

	for i = 2,3 do 
	 	dG[i] 		= torch.zeros(dim[1])
	end
	for t = event_count,2,-1 do
		dG[1]	= criterion:backward(model_table[t].output[1], torch.Tensor(1):fill(targets[t]))
		dG 		= model_table[t]:backward(model_table[t-1].output,dG)
	end	
	dG[1] 		= criterion:backward(model_table[1].output[1], torch.Tensor(1):fill(targets[1]))
	model_table[1]:backward({inputs[1],state_in[1],state_in[2]},dG)
end

function HawkesAvg()
	local events 	= 100000
	local inp,out  	= HawkesNdim(alpha, beta, delta, lambda0, events, no_of_processes)
	local average 	= torch.mean(out)
	return average
end


local function iterate()
	local model_error 	= 0
	local average_error	= 0

	local dt 	= dt0--/event_count

		for j=1,#mdx do
			mdx[j]:zero()
		end
		x_loss,avg_loss = forward_pass()
		backward_pass()
		for j = 1,#mx do
			local Grad = 1.0/(torch.sqrt(#mx)*mdx[j]:norm())

			if Grad ~= Grad then
				print('Gradient normalization constant undefined')
				pause() 
			end

			mx[j]:add(-dt*Grad*mdx[j])
		end

		model_error 	= model_error + x_loss
		average_error 	= average_error + avg_loss

		local relative_error 	= model_error/average_error

	if perp ~= perp then
		print('x_loss = ' .. x_loss)
		print('err = '.. err)
		print('normal iterate blew up')
		pause()
	end
	return 	relative_error, model_error
end 

model 		= LSTMcell()
replicate_module()
average 	= HawkesAvg()
history_new	= torch.zeros(50)
history_old = torch.zeros(50)

local relative_error
local average_error

-- Initializing state table
--[[ 	entry 1 of the following state table corresponds to the hidden state vector
		entry 2 corresponds to the cell state vector
--]]
state_in 	= {[2] = torch.zeros(dim[1]), [1] = torch.zeros(dim[1]), [0] = torch.zeros(dim[0])}
state 		= {[2] = torch.zeros(dim[1]), [1] = torch.zeros(dim[1]), [0] = torch.zeros(dim[0])}
state_out	= {[2] = torch.zeros(dim[1]), [1] = torch.zeros(dim[1]), [0] = torch.zeros(dim[0])}


i = 0
--delta_mx = 1
while dt0>.001 do
	i = i + 1
	inputs,targets = HawkesNdim(alpha, beta, delta, lambda0, event_count,no_of_processes)
	relative_error, model_error = iterate()
	history_new[i%50+1] = model_error

	if i%50 == 0 then 
		new = torch.median(history_new)
		old = torch.median(history_old)
		history_old:copy(history_new)
		print('relative error = '.. relative_error .. '\t absolute error = ' .. model_error)
		if new[1]>.999*old[1] and i>51 then
			dt0 = step_reduction*dt0
		print(' ---------- Step size updated to ' .. dt0 .. '------------')
		end
	end

end 
local final = torch.mean(history_old)
print('---------- Final Results ----------')
print('Total Epochs \t\t\t\t= ' .. i .. '  \nError Relative to the Average \t\t= ' .. relative_error .. ' \nAbsolute Error \t\t\t\t= ' .. model_error .. '\nAverage Error over final 10 epochs \t= '.. final)

return relative_error, model_error, final 

end