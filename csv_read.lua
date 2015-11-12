require 'csvigo'

local csv_utils = {}

function csv_utils.csv_read(csv_path, delimiter, out_mode, start_idx, batch_size)
  X_table = {}
  local count = 1
  local batch_count = 1
  for line in io.lines(csv_path) do
    if batch_size > 0 then 
      if count >= start_idx and count<(start_idx + batch_size) then     
        X_table[batch_count] = line
        batch_count = batch_count + 1
      end
    count = count + 1
    else
      X_table[#X_table+1] = line
    end
  end
  if out_mode == 'table' then
      return X_table
  elseif out_mode == 'tensor' then
    num_instance = #X_table[1]:split(delimiter)
    X_tensor = torch.zeros(#X_table,num_instance)
    for i = 1,#X_table do
      X_tensor[i] = torch.Tensor(X_table[i]:split(delimiter))
    end
    return torch.Tensor(X_tensor)
  else
    error("Incorrect output format")
  end
end

function csv_utils.csv_write(t1,file_path)
 
  local t2 = {}
  for i=1,t1:size(1) do
    t2[i] = {}
    for j=1,t1:size(2) do
      t2[i][j] = t1[i][j]
    end
  end
  csvigo.save{data = t2 ,path = file_path}
end

return csv_utils
    

    
