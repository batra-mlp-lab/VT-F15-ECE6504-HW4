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
fc7_feat = torch.load('fc7_hw4.t7')

lineQA = csv_utils.csv_read('vqa_test.txt', '\n','table',1,-1)
print('Total number of QA instances loaded is',	#lineQA)
imageid_test = {}
question_test = {}
answer_test = {}

-- process text to separate V/Q/A and form vocabulary
for k,v in ipairs(lineQA) do
  for i=1,#v do
    if v:sub(i,i) == ' ' then
      imageid_test[k] = v:sub(1,i-1)
      v = v:sub(i+1,#v)
      break
    end
  end
  -- image id is removed
  local temp = v:split('?')
  question_test[k] = temp[1]:split(' ')
  -- if answers have more than one word, it is pruned to first word
  -- example white and red is pruned to just white! 
  answer_test[k] = temp[2]:split(' ')[1]
end

vocab = data_hw4['question_vocab']
class = data_hw4['answer_map']

vocab_map = invert_table(vocab)
class_map = invert_table(class)

qEn = {}
for k,v in pairs(question_test) do
  tempEn = torch.Tensor(#v)
  for kq,vq in ipairs(v) do
    tempEn[kq] = vocab_map['UNK']
    if table.contains(vocab,vq) then tempEn[kq] = vocab_map[vq] end    
  end
  qEn[k] = tempEn
end

aIn = {}
for k,v in pairs(answer_test) do
  aIn[k] = class_map[v]
end

imIn = torch.IntTensor(#imageid_test)
for k,v in pairs(imageid_test) do
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

print('saving TEST question, answer and image file')
torch.save('data_HW4_test.t7',data_table)


