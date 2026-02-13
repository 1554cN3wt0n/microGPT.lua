--[[
The most atomic way to train and inference a GPT in pure, dependency-free Lua.
This file is the complete algorithm.
Everything else is just efficiency.

Ported from @karpathy's Python original.
--]]

math.randomseed(42) -- Let there be order among chaos

-- ---------------------------------------------------------------------------
-- Utilities
-- ---------------------------------------------------------------------------

local function file_exists(path)
	local f = io.open(path, "r")
	if f then
		f:close()
		return true
	end
	return false
end

local function read_file(path)
	local f = assert(io.open(path, "r"))
	local content = f:read("*a")
	f:close()
	return content
end

local function download_file(url, dest)
	-- Try curl, then wget
	local ok = os.execute(string.format('curl -fsSL "%s" -o "%s"', url, dest))
	if not ok then
		ok = os.execute(string.format('wget -q "%s" -O "%s"', url, dest))
	end
	assert(ok, "Failed to download " .. url .. " — please download it manually.")
end

-- Shuffle a table in-place (Fisher-Yates)
local function shuffle(t)
	for i = #t, 2, -1 do
		local j = math.random(i)
		t[i], t[j] = t[j], t[i]
	end
end

-- Stable sort to get unique characters
local function sorted_unique_chars(str)
	local seen = {}
	local chars = {}
	for i = 1, #str do
		local c = str:sub(i, i)
		if not seen[c] then
			seen[c] = true
			chars[#chars + 1] = c
		end
	end
	table.sort(chars)
	return chars
end

local function table_index(t, val)
	for i, v in ipairs(t) do
		if v == val then
			return i - 1
		end -- 0-based index to match Python
	end
	return nil
end

-- ---------------------------------------------------------------------------
-- Dataset
-- ---------------------------------------------------------------------------

if not file_exists("input.txt") then
	local url = "https://raw.githubusercontent.com/karpathy/makemore/refs/heads/master/names.txt"
	print("Downloading dataset...")
	download_file(url, "input.txt")
end

local docs = {}
for line_ in read_file("input.txt"):gmatch("[^\n]+") do
	line = line_:match("^%s*(.-)%s*$") -- strip
	if #line > 0 then
		docs[#docs + 1] = line
	end
end
shuffle(docs)
print(string.format("num docs: %d", #docs))

-- Build vocabulary from all unique characters
local all_text = table.concat(docs, "")
local uchars = sorted_unique_chars(all_text) -- list of chars, 0-indexed conceptually

-- char -> id helper (0-based)
local char_to_id = {}
for i, c in ipairs(uchars) do
	char_to_id[c] = i - 1 -- 0-based
end

local BOS = #uchars -- BOS token id
local vocab_size = #uchars + 1
print(string.format("vocab size: %d", vocab_size))

-- ---------------------------------------------------------------------------
-- Autograd: Value class
-- ---------------------------------------------------------------------------

local Value = {}
Value.__index = Value

function Value.new(data, children, local_grads)
	return setmetatable({
		data = data,
		grad = 0,
		_children = children or {},
		_local_grads = local_grads or {},
	}, Value)
end

function Value:__add(other)
	if type(other) == "number" then
		other = Value.new(other)
	end
	return Value.new(self.data + other.data, { self, other }, { 1, 1 })
end

function Value:__mul(other)
	if type(other) == "number" then
		other = Value.new(other)
	end
	return Value.new(self.data * other.data, { self, other }, { other.data, self.data })
end

function Value:__pow(other)
	-- other is a plain number
	return Value.new(self.data ^ other, { self }, { other * self.data ^ (other - 1) })
end

function Value:log()
	return Value.new(math.log(self.data), { self }, { 1 / self.data })
end

function Value:exp()
	local e = math.exp(self.data)
	return Value.new(e, { self }, { e })
end

function Value:relu()
	local r = math.max(0, self.data)
	return Value.new(r, { self }, { self.data > 0 and 1.0 or 0.0 })
end

function Value:__unm()
	return self * -1
end

function Value:__sub(other)
	if type(other) == "number" then
		other = Value.new(other)
	end
	return self + -other
end

function Value:__div(other)
	if type(other) == "number" then
		other = Value.new(other)
	end
	return self * (other ^ -1)
end

-- Support number op Value  (e.g.  1/v  or  2*v)
Value.__radd = Value.__add
Value.__rmul = Value.__mul
Value.__rsub = function(self, other)
	if type(other) == "number" then
		other = Value.new(other)
	end
	return other + -self
end
Value.__rdiv = function(self, other)
	if type(other) == "number" then
		other = Value.new(other)
	end
	return other * (self ^ -1)
end

function Value:backward()
	-- Build topological order
	local topo = {}
	local visited = {}
	local function build_topo(v)
		if not visited[v] then
			visited[v] = true
			for _, child in ipairs(v._children) do
				build_topo(child)
			end
			topo[#topo + 1] = v
		end
	end
	build_topo(self)
	self.grad = 1
	for i = #topo, 1, -1 do
		local node = topo[i]
		for j, child in ipairs(node._children) do
			child.grad = child.grad + node._local_grads[j] * node.grad
		end
	end
end

-- ---------------------------------------------------------------------------
-- Model parameters
-- ---------------------------------------------------------------------------

local n_embd = 16
local n_head = 4
local n_layer = 1
local block_size = 8
local head_dim = n_embd / n_head

local function gauss(std)
	-- Box-Muller
	local u1 = math.random()
	local u2 = math.random()
	return std * math.sqrt(-2 * math.log(u1)) * math.cos(2 * math.pi * u2)
end

local function make_matrix(nout, nin, std)
	std = std or 0.02
	local m = {}
	for i = 1, nout do
		m[i] = {}
		for j = 1, nin do
			m[i][j] = Value.new(gauss(std))
		end
	end
	return m
end

local state_dict = {
	wte = make_matrix(vocab_size, n_embd),
	wpe = make_matrix(block_size, n_embd),
	lm_head = make_matrix(vocab_size, n_embd),
}
for i = 0, n_layer - 1 do
	state_dict["layer" .. i .. ".attn_wq"] = make_matrix(n_embd, n_embd)
	state_dict["layer" .. i .. ".attn_wk"] = make_matrix(n_embd, n_embd)
	state_dict["layer" .. i .. ".attn_wv"] = make_matrix(n_embd, n_embd)
	state_dict["layer" .. i .. ".attn_wo"] = make_matrix(n_embd, n_embd, 0)
	state_dict["layer" .. i .. ".mlp_fc1"] = make_matrix(4 * n_embd, n_embd)
	state_dict["layer" .. i .. ".mlp_fc2"] = make_matrix(n_embd, 4 * n_embd, 0)
end

-- Flatten all parameters into a single list
local params = {}
for _, mat in pairs(state_dict) do
	for _, row in ipairs(mat) do
		for _, p in ipairs(row) do
			params[#params + 1] = p
		end
	end
end
print(string.format("num params: %d", #params))

-- ---------------------------------------------------------------------------
-- Model forward pass helpers
-- ---------------------------------------------------------------------------

local function linear(x, w)
	-- x: list[Value], w: list[list[Value]]  ->  list[Value]
	local out = {}
	for _, wo in ipairs(w) do
		local acc = Value.new(0)
		for j, wij in ipairs(wo) do
			acc = acc + wij * x[j]
		end
		out[#out + 1] = acc
	end
	return out
end

local function softmax(logits)
	local max_val = logits[1].data
	for i = 2, #logits do
		if logits[i].data > max_val then
			max_val = logits[i].data
		end
	end
	local exps = {}
	for _, val in ipairs(logits) do
		exps[#exps + 1] = (val - max_val):exp()
	end
	local total = exps[1]
	for i = 2, #exps do
		total = total + exps[i]
	end
	local out = {}
	for _, e in ipairs(exps) do
		out[#out + 1] = e / total
	end
	return out
end

local function rmsnorm(x)
	local ms = Value.new(0)
	for _, xi in ipairs(x) do
		ms = ms + xi * xi
	end
	ms = ms * (1 / #x)
	local scale = (ms + 1e-5) ^ -0.5
	local out = {}
	for _, xi in ipairs(x) do
		out[#out + 1] = xi * scale
	end
	return out
end

-- ---------------------------------------------------------------------------
-- GPT forward: token_id and pos_id are 0-based integers
-- keys, values are tables of layers, each containing accumulated k/v lists
-- ---------------------------------------------------------------------------

local function gpt(token_id, pos_id, keys, values)
	-- Lua tables are 1-indexed, so shift by +1 when indexing state_dict rows
	local tok_emb = state_dict["wte"][token_id + 1]
	local pos_emb = state_dict["wpe"][pos_id + 1]
	local x = {}
	for i = 1, n_embd do
		x[i] = tok_emb[i] + pos_emb[i]
	end
	x = rmsnorm(x)

	for li = 0, n_layer - 1 do
		-- 1) Multi-head attention
		local x_residual = x
		x = rmsnorm(x)
		local q = linear(x, state_dict["layer" .. li .. ".attn_wq"])
		local k = linear(x, state_dict["layer" .. li .. ".attn_wk"])
		local v = linear(x, state_dict["layer" .. li .. ".attn_wv"])
		local layer_keys = keys[li + 1]
		local layer_values = values[li + 1]
		layer_keys[#layer_keys + 1] = k
		layer_values[#layer_values + 1] = v
		local x_attn = {}
		for h = 0, n_head - 1 do
			local hs = h * head_dim
			local q_h = {}
			for j = 1, head_dim do
				q_h[j] = q[hs + j]
			end
			local k_h = {}
			for t = 1, #layer_keys do
				k_h[t] = {}
				for j = 1, head_dim do
					k_h[t][j] = layer_keys[t][hs + j]
				end
			end
			local v_h = {}
			for t = 1, #layer_values do
				v_h[t] = {}
				for j = 1, head_dim do
					v_h[t][j] = layer_values[t][hs + j]
				end
			end
			-- Attention logits
			local attn_logits = {}
			for t = 1, #k_h do
				local acc = Value.new(0)
				for j = 1, head_dim do
					acc = acc + q_h[j] * k_h[t][j]
				end
				attn_logits[t] = acc * (head_dim ^ -0.5)
			end
			local attn_weights = softmax(attn_logits)
			-- Weighted sum of values
			for j = 1, head_dim do
				local acc = Value.new(0)
				for t = 1, #v_h do
					acc = acc + attn_weights[t] * v_h[t][j]
				end
				x_attn[hs + j] = acc
			end
		end
		x = linear(x_attn, state_dict["layer" .. li .. ".attn_wo"])
		for i = 1, n_embd do
			x[i] = x[i] + x_residual[i]
		end

		-- 2) MLP block
		x_residual = x
		x = rmsnorm(x)
		x = linear(x, state_dict["layer" .. li .. ".mlp_fc1"])
		for i = 1, #x do
			local r = x[i]:relu()
			x[i] = r ^ 2
		end
		x = linear(x, state_dict["layer" .. li .. ".mlp_fc2"])
		for i = 1, n_embd do
			x[i] = x[i] + x_residual[i]
		end
	end

	local logits = linear(x, state_dict["lm_head"])
	return logits
end

-- ---------------------------------------------------------------------------
-- Adam optimizer buffers
-- ---------------------------------------------------------------------------

local learning_rate = 1e-2
local beta1 = 0.9
local beta2 = 0.95
local eps_adam = 1e-8

local m_buf = {}
local v_buf = {}
for i = 1, #params do
	m_buf[i] = 0.0
	v_buf[i] = 0.0
end

-- ---------------------------------------------------------------------------
-- Training loop
-- ---------------------------------------------------------------------------

local num_steps = 500

for step = 1, num_steps do
	local doc = docs[((step - 1) % #docs) + 1]

	-- Tokenize: BOS + char ids + BOS  (all 0-based token ids)
	local tokens = { BOS }
	for i = 1, #doc do
		tokens[#tokens + 1] = char_to_id[doc:sub(i, i)]
	end
	tokens[#tokens + 1] = BOS

	local n = math.min(block_size, #tokens - 1)

	-- Forward pass
	local keys = {}
	local values_ = {}
	for li = 1, n_layer do
		keys[li] = {}
		values_[li] = {}
	end

	local losses = {}
	for pos_id = 0, n - 1 do
		local token_id = tokens[pos_id + 1]
		local target_id = tokens[pos_id + 2]
		local logits = gpt(token_id, pos_id, keys, values_)
		local probs = softmax(logits)
		-- target_id is 0-based, probs is 1-indexed
		local loss_t = -(probs[target_id + 1]:log())
		losses[#losses + 1] = loss_t
	end

	-- Average loss
	local loss_sum = losses[1]
	for i = 2, #losses do
		loss_sum = loss_sum + losses[i]
	end
	local loss = loss_sum * (1 / n)

	-- Backward
	loss:backward()

	-- Cosine LR decay
	local lr_t = learning_rate * 0.5 * (1 + math.cos(math.pi * (step - 1) / num_steps))

	-- Adam update
	for i, p in ipairs(params) do
		m_buf[i] = beta1 * m_buf[i] + (1 - beta1) * p.grad
		v_buf[i] = beta2 * v_buf[i] + (1 - beta2) * p.grad ^ 2
		local m_hat = m_buf[i] / (1 - beta1 ^ step)
		local v_hat = v_buf[i] / (1 - beta2 ^ step)
		p.data = p.data - lr_t * m_hat / (math.sqrt(v_hat) + eps_adam)
		p.grad = 0
	end

	print(string.format("step %4d / %4d | loss %.4f", step, num_steps, loss.data))
end

-- ---------------------------------------------------------------------------
-- Inference
-- ---------------------------------------------------------------------------

local temperature = 0.5

print("\n--- inference ---")

local function weighted_choice(weights)
	local total = 0
	for _, w in ipairs(weights) do
		total = total + w
	end
	local r = math.random() * total
	local cumulative = 0
	for i, w in ipairs(weights) do
		cumulative = cumulative + w
		if r <= cumulative then
			return i
		end
	end
	return #weights
end

for sample_idx = 1, 20 do
	local keys_inf = {}
	local values_inf = {}
	for li = 1, n_layer do
		keys_inf[li] = {}
		values_inf[li] = {}
	end

	local token_id = BOS
	local sample = {}

	for pos_id = 0, block_size - 1 do
		local logits = gpt(token_id, pos_id, keys_inf, values_inf)
		-- Apply temperature: divide each logit (Value) by temperature scalar
		local scaled = {}
		for _, l in ipairs(logits) do
			scaled[#scaled + 1] = l * (1 / temperature)
		end
		local probs = softmax(scaled)
		local weights = {}
		for _, p in ipairs(probs) do
			weights[#weights + 1] = p.data
		end
		token_id = weighted_choice(weights) - 1 -- back to 0-based
		if token_id == BOS then
			break
		end
		sample[#sample + 1] = uchars[token_id + 1]
	end

	print(string.format("sample %2d: %s", sample_idx, table.concat(sample, "")))
end
