# ES + IK：IK 版本必须与 ES 完全一致，换 ES 版本时 IK 要一起换
ARG ES_VERSION=8.19.23
FROM docker.elastic.co/elasticsearch/elasticsearch:${ES_VERSION}
ARG ES_VERSION

RUN bin/elasticsearch-plugin install --batch \
      "https://get.infini.cloud/elasticsearch/analysis-ik/${ES_VERSION}"

# IK 的配置目录以安装后的实际位置为准（8.x 为 config/analysis-ik）。
# 把扩展词典指向 custom/ 子目录，该目录由 compose bind mount 到仓库的 docker/elasticsearch/ik-dict/。
ENV IK_CONF_DIR=/usr/share/elasticsearch/config/analysis-ik
RUN test -f "${IK_CONF_DIR}/IKAnalyzer.cfg.xml" \
 && sed -i 's#<entry key="ext_dict">[^<]*</entry>#<entry key="ext_dict">custom/novel_ext.dic</entry>#' \
      "${IK_CONF_DIR}/IKAnalyzer.cfg.xml" \
 && grep -q 'custom/novel_ext.dic' "${IK_CONF_DIR}/IKAnalyzer.cfg.xml" \
 && mkdir -p "${IK_CONF_DIR}/custom"
