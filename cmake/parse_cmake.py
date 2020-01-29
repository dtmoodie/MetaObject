import argparse
import os

sections =[
    'project_id:',
    'include_dirs:',
    'lib_dirs_debug:',
    'lib_dirs_release:',
    'compile_options:',
    'compile_definitions:',
    'module_dependencies:',
    'compiler_location:',
    'link_libs:',
    'link_libs_debug:',
    'link_libs_release:'
]

parser = argparse.ArgumentParser()

parser.add_argument("in_path", type=str)
parser.add_argument("out_path", type=str)
parser.add_argument("plugin_name", type=str)
args = parser.parse_args()

project_id = 0

def removeDuplicates(x):
    out = []
    for i in x:
        if i not in out:
            out.append(i)
    return out

def parseSection(start_index, lines):
    j = start_index + 1
    for j in range(j, len(lines)):
        if lines[j].strip() in sections:
            break
    include_dirs = lines[start_index+1:j]
    out = []
    for x in include_dirs:
        x = [y.strip() for y in x.split(';')]
        for y in x:
            if(len(y)):
                out.append(y)
    return removeDuplicates(out)

def parsePathSection(start_index, lines):
    for j in range(start_index+1, len(lines)):
        if lines[j].strip() in sections:
            break
    include_dirs = lines[start_index+1:j]
    out = []
    for x in include_dirs:
        x = x.strip()
        if(os.path.exists(x)):
            out.append(x)
    return removeDuplicates(out)

def writeList(out, paths, prefix = '', postfix = ''):
    out.write('{\n')
    out.write('    static const char* paths[] = {')
    if(len(paths) == 0):
        out.write('\n        nullptr\n')
    else:
        for i, inc in enumerate(paths):
            if( i != 0):
                out.write(',')
            out.write('\n        "{}{}{}"'.format(prefix, inc, postfix))
        out.write(',\n        nullptr')
    out.write('\n    };\n')
    out.write('    return paths;\n')

with open(args.in_path, 'rt') as f:
    with open(args.out_path, 'wt') as out:
        lines = f.readlines()
        for i in range(len(lines)):
            if('project_id' in lines[i]):
                project_id = lines[i+1]
            if('include_dirs' in lines[i]):
                include_dirs = parsePathSection(i, lines)
            if('lib_dirs_debug' in lines[i]):
                debug_dirs = parsePathSection(i, lines)
            if('lib_dirs_release' in lines[i]):
                release_dirs = parsePathSection(i, lines)
            if('compile_options' in lines[i]):
                compile_opts = parseSection(i, lines)
                compile_opts = [x if 'c++11' not in x else x.replace('c++11','gnu++14') for x in compile_opts]
            if('compile_definitions' in lines[i]):
                compile_defs = parseSection(i, lines)
            if('module_dependencies' in lines[i]):
                deps = parseSection(i, lines)
            if('compiler_location' in lines[i]):
                cloc = parseSection(i, lines)
            if('link_libs:' in lines[i]):
                link_libs = parseSection(i, lines)
            if('link_libs_debug' in lines[i]):
                link_libs_debug = parseSection(i, lines)
            if('link_libs_release' in lines[i]):
                link_libs_release = parseSection(i, lines)

        out.write('#include "{}_export.hpp"\n\n'.format(args.plugin_name))

        out.write('const char** getPluginIncludes()\n')
        writeList(out, include_dirs)
        out.write('}\n\n')

        out.write('const char** getPluginLinkDirsDebug()\n')
        writeList(out, debug_dirs)
        out.write('}\n\n')

        out.write('const char** getPluginLinkDirsRelease()\n')
        writeList(out, release_dirs)
        out.write('}\n\n')

        out.write('const char** getPluginCompileOptions()\n')
        writeList(out, compile_opts, postfix=' ')
        out.write('}\n\n')

        out.write('const char** getPluginCompileDefinitions()\n')
        writeList(out, compile_defs + ['{}_EXPORTS'.format(args.plugin_name)], prefix='-D', postfix=' ')
        out.write('}\n\n')

        out.write('const char** getPluginLinkLibs()\n')
        writeList(out, link_libs, prefix='-l', postfix=' ')
        out.write('}\n\n')

        out.write('const char** getPluginLinkLibsDebug()\n')
        writeList(out, link_libs_debug, prefix='-l', postfix=' ')
        out.write('}\n\n')

        out.write('const char** getPluginLinkLibsRelease()\n')
        writeList(out, link_libs_release, prefix='-l', postfix=' ')
        out.write('}\n\n')

        out.write('int getPluginProjectId()\n')
        out.write('{\n')
        out.write('    return {};\n'.format(project_id))
        out.write('}\n')


