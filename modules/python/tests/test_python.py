import metaobject as mo

mo.plugins.loadPlugins('./bin/Plugins')

obj = mo.object.AddBinary()

print(len(obj.getInputs()))
