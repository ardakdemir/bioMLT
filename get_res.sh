pref=${1}
for i in 1 2 3 4 5 
do
cat $pref${i}".txt" | tail -2 | head -1 > out"_"${i}.txt
done
python get_average.py out_
