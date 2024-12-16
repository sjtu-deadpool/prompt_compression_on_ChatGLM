import os  

def print_file_structure(start_path, indent=0):  
    for item in os.listdir(start_path):  
        item_path = os.path.join(start_path, item)  
        print('  ' * indent + f'|- {item}')  
        if os.path.isdir(item_path):  
            print_file_structure(item_path, indent + 1)  

# 打印当前文件夹的文件结构  
print_file_structure('.')