# Todo

- [x] Rewrite training as generator
  - [x] Test implementation
- [x] Implement callback for test evaluation on test set
- [x] Run implementation
  - No success
- [x] Implement other matches
  - [x] MaxPool
  - [x] Attentive
  - [x] MaxAttentive
- [x] Rewrite implementation to use GPU.
  - Solved: Running with CUDA_VISIBLE_DEVICES=$1
- [ ] Debug why compilation of the model is so slow.. Di-sect which layer causes it (probably due to matching layers and looping 88 times pr batch.)
- [ ] Try increasing batch_size, don't care about memory. we got lots of that
- [ ] Shuffle input files
- [ ] 2nd run of implementation
  - [ ] Add dropout
  - [ ] Reduce training set, and experiment
  - [ ] Run
