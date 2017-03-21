# Todo

1. Implement other matches
2. Rewrite implementation to use GPU with sessions i recon?
3. Rewrite training as generator [ala](https://github.com/fchollet/keras/issues/2708#issuecomment-218781778)
```python
def generate_arrays_from_file(path):
  while 1:
    f = open(path)
    for line in f
      # create numpy arrays of input data
      # and labels, from each line in the file
      x, y = process_line(line)
      img = load_images(x)
      yield (img, y)
    f.close()
model.fit_generator(generate_arrays_from_file('/my_file.txt'), samples_per_epoch=10000, nb_epoch=10)  
```
