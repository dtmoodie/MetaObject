import metaobject as mo

print(len(obj.getInputs()))
params = obj.getParams()
print(params[1].data)
params[1].data = [(0,0,1,1)]
rect = params[1].data[0]

if(rect.x != 0.0):
    print('Rect x value not set')
if(rect.y != 0.0):
    print('Rect y value not set')

if(rect.width != 1.0):
    print('Rect width value not set')

if(rect.height != 1.0):
    print('Rect height value not set')

params[1].data = [(0,0,1,1), (0.25,0.25,0.5,0.5)]

rect = params[1].data[1]

if(rect.x != 0.25):
    print('Rect x value not set')
if(rect.y != 0.25):
    print('Rect y value not set')

if(rect.width != 0.5):
    print('Rect width value not set')

if(rect.height != 0.5):
    print('Rect height value not set')
