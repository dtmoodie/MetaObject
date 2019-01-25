import metaobject as mo

assert(mo.plugins.loadPlugin('libmo_objectplugind.so'))
print(mo.listConstructableObjects())

obj = mo.object.SerializableObject()

assert(obj.test == 5)
assert(obj.test2 == 6)

obj.test = 10
obj.test2 = 20

assert(obj.test == 10)
assert(obj.test2 == 20)

obj = mo.object.DerivedSignals()

assert(obj.base_param == 5)
