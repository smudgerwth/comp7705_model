import pandas as pd
from sdv.single_table import GaussianCopulaSynthesizer  # 新版本中的类名
from sdv.metadata import SingleTableMetadata

# 1. 加载原始数据
df_original = pd.read_csv("synthetic_health_data.csv")

# 2. 定义元数据（可选，但推荐）
metadata = SingleTableMetadata()
metadata.detect_from_dataframe(df_original)

# 3. 初始化并训练模型
synthesizer = GaussianCopulaSynthesizer(metadata)
synthesizer.fit(df_original)

# 4. 生成新数据
df_new = synthesizer.sample(num_rows=200000)

# 5. 保存
df_new.to_csv("synthetic_expanded_health_data_20k.csv", index=False)
print("数据已成功生成并保存！")
