#!@Python_EXECUTABLE@

import metaobject as mo
import glob

plugin = glob.glob('@PLUGIN_PATH@/libmo_objectplugin*')
assert(mo.plugins.loadPlugin(plugin[0]))
print(mo.listConstructableObjects())

obj = mo.object.SerializableObject()

assert(obj.test.data == 5)
assert(obj.test2.data == 6)

obj.test = 10
obj.test2 = 20

assert obj.test.data == 10, "Failed to set value"
assert(obj.test2.data == 20)

obj = mo.object.DerivedSignals()

assert(obj.base_param.data == 5)

types = mo.listConstructableObjects()

pt = mo.datatypes.Point2d(x=1, y=2)

assert(pt.x == 1.0)
assert(pt.y == 2.0)
pt.x = 2.0
pt.y = 3.0
assert(pt.x == 2.0)
assert(pt.y == 3.0)

print('Success')
