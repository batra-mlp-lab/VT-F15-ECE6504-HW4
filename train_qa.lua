require 'rnn'
hw4_util = require 'hw4_utils'

cmd = torch.CmdLine()
cmd:text()
cmd:text('Train an LSTM QA model')
cmd:option('--learningRate', 0.05, 'learning rate at t=0')
cmd:option('--momentum', 0.9, 'momentum')
cmd:option('--batchSize', 32, 'momentum')
cmd:option('--cuda', false, 'use CUDA')
cmd:option('--useDevice', 1, 'sets the device (GPU) to use')
cmd:option('--maxEpoch', 1000, 'maximum number of epochs to run')
cmd:option('--testInter', 100, 'test after T iterations')
cmd:option('--disp', 64, 'display loss every disp iterations')
cmd:option('--hiddenSize', 64, 'maximum number of epochs to run')
cmd:option('--batchSize', 32, 'batch size')
cmd:option('--dropout', false, 'apply dropout after each recurrent layer')
cmd:option('--saveEvery', 500 , 'saves after S iterations')
cmd:text()
opt = cmd:parse(arg or {})
table.print(opt)

if opt.cuda == true then
  require 'cunn'
  require 'cutorch'
  cutorch.setDevice(opt.useDevice)
end

-- load data
data_hw4 = torch.load('data_HW4.t7' )
-- contains
-- questions [encoded as per question_vocab]
-- answers [encoded question_vocab]
-- question_vocab - vocabulary: idx ->words - any word with occurence less than 3 in the train set is replaced with UNK token which is the last word in the vocabulary
-- answer_map: idx -> class label
question = data_hw4['question'] 
answer = data_hw4['answer']
image = data_hw4['image']

-- batchify
-- data is put in a table of batchSize X Q_len 
X_train,y_train,_ = hw4_util.batchify(question,answer,image,opt.batchSize)

-- separate some part of the data into val
val_split = 95 -- approx. 20% of the data used for val
X_val = {}
y_val = {}
for i=1,val_split do
  X_val[i] = X_train[i]
  y_val[i] = y_train[i]
  table.remove(X_train,i)
  table.remove(y_train,i)
end

-- define model
vocab_size = #data_hw4['question_vocab']
nClasses = #data_hw4['answer_map']

------------MY MODEL-----------------
model = nn.Identity()
-- Each word is associated with a distributed representation. This can be done
-- by using a LookupTable. These can either be initialized randomly or with word 
-- embeddings like word2vec. 
------MODEL DEF STARTS---------------
-------------------------------------
-------------------------------------
-- your model goes here
-------------------------------------
-------------------------------------
model:add(nn.Linear(opt.hiddenSize,nClasses)) -- W^Tx
model:add(nn.LogSoftMax()) 
------ MODEL DEF ENDS----------------

-- cross entropy loss. Add LogSoftMax layer at the end of the layer
criterion = nn.ClassNLLCriterion()

if opt.cuda == true then 
  model:cuda() 
  criterion:cuda()
end

-- making sure
model:reset()
model:zeroGradParameters()

-- training
for epoch = 1,opt.maxEpoch do
  for i=1,#X_train do
    iter = (epoch-1)*#question + i
    x = X_train[i]
    y = y_train[i]
    if opt.cuda == true then
      x = x:cuda()
      y = y:cuda()
    end
    local p = model:forward(x)
    local err = criterion:forward(p,y)
    model:zeroGradParameters()
    local g = criterion:backward(p,y)
    model:backward(x,g)
    model:updateGradParameters(opt.momentum) -- SGD with momentum is implemented here. Other optimizations can be used via 'optim' library
    model:updateParameters(opt.learningRate)
    model:zeroGradParameters()
    if iter%opt.disp == 0 then
      print('epoch:', epoch)
      print('iter', iter)
      print('training loss', err)
    end  
    if iter%opt.testInter == 0 then
      model:evaluate()
      test_err = torch.zeros(#X_val)
      for test_i = 1,#X_val do
        x = X_val[test_i]
        y = y_val[test_i]
        if opt.cuda == true then
          x = x:cuda()
          y = y:cuda()
        end
        local p = model:forward(x)
        test_err[test_i] = criterion:forward(p,y)        
      end 
      print('epoch:', epoch)
      print('iter', iter)
      print('testing loss', test_err:mean())            
      model:training()
    end
    if iter%opt.saveEvery == 0 then
      torch.save('model_' .. tostring(epoch) .. '_' .. tostring(iter) .. '.t7' , model)
    end
  end
end
