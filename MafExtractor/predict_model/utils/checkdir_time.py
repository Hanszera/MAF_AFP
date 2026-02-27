import os
from datetime import datetime



def create_numbered_subfolder(base_path,num=0,is_check_point=False):

    try:
        # 获取当前年月日，格式为YYYY-MM-DD
        current_date = datetime.now().strftime("%Y-%m-%d")

        # 构建日期文件夹的完整路径
        date_folder_path = os.path.join(base_path, current_date)

        # 检查基础路径是否存在
        if not os.path.exists(base_path):
            raise FileNotFoundError(f"基础路径不存在: {base_path}")

        if not is_check_point:
            if not os.path.exists(date_folder_path):
                # 如果日期文件夹不存在，创建它
                os.makedirs(date_folder_path)
                # 创建第一个子文件夹（名为1）
                new_subfolder_number = 1
                new_subfolder_path = os.path.join(date_folder_path, str(new_subfolder_number))
                os.makedirs(new_subfolder_path)
                return new_subfolder_path

            else:

                all_items = os.listdir(date_folder_path)


                subfolders = []
                for item in all_items:
                    item_path = os.path.join(date_folder_path, item)
                    if os.path.isdir(item_path):
                        subfolders.append(item)


                subfolder_count = len(subfolders)


                new_subfolder_number = subfolder_count + 1
                new_subfolder_path = os.path.join(date_folder_path, str(new_subfolder_number))
                print(f'{new_subfolder_number}times_run...')

                os.makedirs(new_subfolder_path)
                return new_subfolder_path
        else:


            new_subfolder_number = num
            new_subfolder_path = os.path.join(date_folder_path, str(new_subfolder_number))
            print(f'{new_subfolder_number}times_run...')
            return new_subfolder_path

    except Exception as e:
        print(f"错误: 创建文件夹时发生异常 - {str(e)}")
        return None


# 使用示例
if __name__ == "__main__":
    # 指定要检查的基础路径（可以是绝对路径或相对路径）
    base_path = "./data"  # 当前目录下的data文件夹

    # 调用函数
    result = create_numbered_subfolder(base_path)

    if result:
        print(f"成功创建文件夹: {result}")
    else:
        print("文件夹创建失败")