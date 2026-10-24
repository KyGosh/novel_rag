# IK 自定义词典

本目录 bind mount 到容器内 `config/analysis-ik/custom/`，`IKAnalyzer.cfg.xml` 的 `ext_dict` 指向 `custom/novel_ext.dic`。

- 每行一个词，UTF-8 无 BOM。P3 从知识图谱导出实体名和别名回灌到这里（带间隔号和不带间隔号的写法都收录）。
- IK 只在启动时加载本地词典：改完要 `docker compose restart elasticsearch`，然后**重建索引**（已索引的文档不会重新分词）。
- `*.dic` 由流程生成、可能含书籍专属词汇，不入库。
