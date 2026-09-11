# Raw data

原始数据不可修改。服务器当前原始题库路径：

```text
/home/share_ssd_data/nfs-data1/wangmeng148/data/tiku/high-geo-hist-pol/生物.jsonl
```

服务器项目同级数据目录约定：

```text
/local_data/zhangyonglin/data/bio-know-tag/
```

大文件默认不提交 Git。复制时使用不会覆盖已有文件的命令，并记录文件行数与哈希；所有清洗过程先写入 `runtime/<timestamp>/`，验收后再物化到上述数据目录。

