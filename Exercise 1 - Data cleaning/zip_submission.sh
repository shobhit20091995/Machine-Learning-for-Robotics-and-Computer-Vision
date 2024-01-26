while getopts p: flag
do
    case "${flag}" in
        p) path=${OPTARG};;
        esac
done


zip -r ex1_mlrcv_submission.zip mlrcv/*.py  ex1_data_clean.ipynb