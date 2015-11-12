npy4th = require 'npy4th'
csv_utils = require 'csv_read'

-- some functions lazily added here
function table.contains(table, element)
  for _, value in pairs(table) do
    if value == element then
      return true
    end
  end
  return false
end

function invert_table(table)
  inv_table = {}
  for k,v in pairs(table) do
    inv_table[v] = k
  end
  return inv_table
end

function compare(a,b)
  return a[1] < b[1]
end

-----------------------------------------------------------------

-- load fc7 features from npy and save it as torch readable 
fc7_what_color = npy4th.loadnpy('fc7_what_color.npy')
ids = csv_utils.csv_read('imgId.csv' ,',','table',1,-1)
-- read fc7 features and form table
fc7_table = {}
for i=1,fc7_what_color:size()[1] do
  fc7_table[tonumber(ids[i])] = fc7_what_color[i]
end

torch.save('fc7_what_color.t7',fc7_table)

lineQA = csv_utils.csv_read('testvqa.txt', '\n','table',1,-1)
print('Total number of QA instances loaded is',	#lineQA)

imageid = {}
question = {}
answer = {}
class = {}
vocab = {}
vocab_count = {}
answer_count = {}
-- process text to separate V/Q/A and form vocabulary
for k,v in ipairs(lineQA) do
  for i=1,#v do
    if v:sub(i,i) == ' ' then
      imageid[k] = v:sub(1,i-1)
      v = v:sub(i+1,#v)
      break
    end
  end
  -- image id is removed
  local temp = v:split('?')
  question[k] = temp[1]:split(' ')
  -- if answers have more than one word, it is pruned to first word
  -- example white and red is pruned to just white! 
  answer[k] = temp[2]:split(' ')[1]
  -- process question and answer entries for new words
  for kq,vq in ipairs(question[k]) do
    -- remove unwanted "" and " " -- bad way of doing things! 
    if vq == "" then table.remove(question[k],kq) end
    if vq == " " then table.remove(question[k],kq) end
    if not table.contains(vocab,vq) then
      vocab[#vocab +1] = vq  
      vocab_count[vq] = 0
    end
    vocab_count[vq] = vocab_count[vq] + 1 
  end
  -- count the classes
  if not table.contains(class,answer[k]) then class[#class+1] = answer[k] end  
end
-- prune vocab for entries with less than 3 counts
for k,v in ipairs(vocab) do
  if vocab_count[v] < 3 then table.remove(vocab,k) end
end
-- add unknown token
vocab[#vocab+1] = 'UNK' 
-- create vocab mapping -- inverting the vocab can be useful when looking up. 
vocab_map = invert_table(vocab)
class_map = invert_table(class)

-- convert question and answer tables into number encodings using the vocab and class maps
qEn = {}
for k,v in pairs(question) do
  tempEn = torch.Tensor(#v)
  for kq,vq in ipairs(v) do
    tempEn[kq] = vocab_map['UNK']
    if table.contains(vocab,vq) then tempEn[kq] = vocab_map[vq] end    
  end
  qEn[k] = tempEn
end

aIn = {}
for k,v in pairs(answer) do
  aIn[k] = class_map[v]
end

-- convert imageids from strings to int
imIn = torch.IntTensor(#imageid)
for k,v in pairs(imageid) do
  imIn[k] = tonumber(v)
end

-- convert into equal length batches
qEn2= {}
unk = #vocab
a10 = torch.range(1,10):long()
for k,v in pairs(qEn) do
  if v:nElement() < 10 then -- pad zero
    tempcat = torch.mul(torch.ones(10 - v:nElement()),unk)
    qEn2[k] = torch.cat(tempcat,torch.Tensor(v),1)
  else 
    qEn2[k] = qEn[k]:index(1,a10)
  end
end

data_table = {}
data_table['question'] = qEn2
data_table['answer'] = aIn
data_table['image'] = imIn
data_table['question_vocab'] = vocab
data_table['answer_map'] = class

print('saving vocab, question, answer and image files')
torch.save('data_HW4.t7',data_table)









      
  


