整个父文件夹放在SDK/${TIDL_INSTALL_PATH}/ti_dl/test/testvecs/下

设置环境变量(根据sdk版本修改):
export TIDL_INSTALL_PATH=/home/wyj/sda2/TAD4VL_SKD_8_6/ti-processor-sdk-rtos-j721s2-evm-08_06_01_03/tidl_j721s2_08_06_00_10

运行：
cd ${TIDL_INSTALL_PATH}/ti_dl/utils/tidlModelImport && ./out/tidl_model_import.out ${TIDL_INSTALL_PATH}/ti_dl/test/testvecs/yolox_importer/tidl_import_seed.txt

