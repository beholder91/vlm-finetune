def extract_all_pdfs(tar_dir, output_dir):
    """
    将所有PDF tar.gz分块文件解压到指定目录
    
    参数:
        tar_dir (str): 包含tar.gz分块文件的目录路径
        output_dir (str): 解压文件的输出目录路径
    
    返回:
        int: 成功解压的文件数量
    """
    import os
    import tarfile
    from tqdm import tqdm
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取所有tar.gz文件
    tar_files = [f for f in os.listdir(tar_dir) if f.endswith('.tar.gz') and f.startswith('pdf_chunk_')]
    
    extracted_count = 0
    
    # 遍历并解压每个tar文件
    for tar_file in tqdm(tar_files, desc="解压PDF文件"):
        tar_path = os.path.join(tar_dir, tar_file)
        try:
            with tarfile.open(tar_path, 'r:gz') as tar:
                # 解压所有文件
                tar.extractall(path=output_dir)
                # 计算本次解压文件数
                pdf_count = sum(1 for member in tar.getmembers() if member.name.endswith('.pdf'))
                extracted_count += pdf_count
                print(f"已从 {tar_file} 解压 {pdf_count} 个PDF文件")
        except Exception as e:
            print(f"解压 {tar_file} 时出错: {e}")
    
    print(f"解压完成，共解压 {extracted_count} 个PDF文件到 {output_dir}")
    return extracted_count

if __name__ == "__main__":
    # 将所有tar.gz文件解压到pdfs目录
    tar_directory = "pdf_tarballs"
    output_directory = "pdfs"
    extract_all_pdfs(tar_directory, output_directory)