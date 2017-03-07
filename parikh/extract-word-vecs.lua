require 'nn'
require 'nngraph'
require 'hdf5'

cmd = torch.CmdLine()
cmd:text("")
cmd:text("**Testing**")
cmd:text("")
cmd:option('-model_file','result.model_final.t7', [[Path to the final model *.t7 file]])
cmd:option('-word_dict', 'entailment.word.dict', [[Path to source vocabulary (*.src.dict file)]])
cmd:option('-output_file1', 'word_vecs_1.txt', [[Path to output the first word embedding]])
cmd:option('-output_file2', 'word_vecs_2.txt', [[Path to output the first word embedding]])
opt = cmd:parse(arg)

function idx2key(file)   
   local f = io.open(file,'r')
   local t = {}
   for line in f:lines() do
	  local c = {}
	  for w in line:gmatch'([^%s]+)' do
	 table.insert(c, w)
	  end
	  t[tonumber(c[2])] = c[1]
   end   
   return t
end

function flip_table(u)
   local t = {}
   for key, value in pairs(u) do
	  t[value] = key
   end
   return t   
end

function sent2wordidx(sent, word2idx, start_symbol)
   local t = {}
   local u = {}
   table.insert(t, START)
   for word in sent:gmatch'([^%s]+)' do
	  local idx = word2idx[word] or UNK 
	  table.insert(t, idx)
   end
   return torch.LongTensor(t)
end

function wordidx2sent(sent, idx2word)
   local t = {}
   for i = 1, sent:size(1) do -- skip START and END
	 table.insert(t, idx2word[sent[i]])	 
   end
   return table.concat(t, ' ')
end

function main()
	opt = cmd:parse(arg)

	print('loading ' .. opt.model_file .. '...')
	checkpoint = torch.load(opt.model_file)
	print('done!')
	model, model_opt = table.unpack(checkpoint)  
	for i = 1, #model do
    	model[i]:evaluate()
	end

	word_vecs_enc1 = model[1]
  	word_vecs_enc2 = model[2]

	idx2word = idx2key(opt.word_dict)
	word2idx = flip_table(idx2word)

	local out_file1 = io.open(opt.output_file1,'w')
	local out_file2 = io.open(opt.output_file2,'w')
	split = ","

	for word, idx in word2idx do
		print("writing word: ")
		print(word)
		output_file1:write(word, split)
		output_file2:write(word, split)
		vec_1 = word_vecs_enc1.weight[idx]
		vec_2 = word_vecs_enc2.weight[idx]
		for i = 1, 300 do
			output_file1:write(vec_1[i], split)
			output_file2:write(vec_2[i], split)
		end
		output_file1:write("\n")
		output_file2:write("\n")
	end
	io.close(output_file1)
	io.close(output_file2)

end
main()
