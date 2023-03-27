import pickle

file = 'tags.pkl'
objects = []
with (open(file, 'rb')) as openfile:
    while True:
        try:
            objects.append(pickle.load(openfile))
        except EOFError:
            break

print(objects)
print('Done')
