hw4_util = {}

function hw4_util.batchify(X,y,image,batch_size)
  X_temp = {}
  y_temp = {}
  I_temp = {}
  I = {}
  for i=1,image:size(1) do
    I[i] = image[i]
  end
  n_batch = torch.ceil(#X/batch_size)
  n_count = 0
  for batch = 1,n_batch do
    X_temp[batch] = {}
    y_temp[batch] = {}  
    I_temp[batch] = {}  
    for batch_count = 1,batch_size do
      table.insert(X_temp[batch], X[n_count%#X + 1])
      table.insert(y_temp[batch], y[n_count%#y + 1])
      table.insert(I_temp[batch], I[n_count%#I + 1])
      n_count = n_count + 1
    end
  end
  Xt = {}
  yt = {}
  It = {}
  for i=1,n_batch do
    x_t = torch.zeros(batch_size,X_temp[1][1]:nElement())
    y_t = torch.zeros(batch_size)
    i_t = torch.zeros(batch_size)
    for j=1,batch_size do
      x_t[j] = X_temp[i][j]
      y_t[j] = y_temp[i][j]
      i_t[j] = I_temp[i][j]
    end
  Xt[i] = x_t
  yt[i] = y_t
  It[i] = i_t
  end
  return Xt,yt,It
end

return hw4_util

