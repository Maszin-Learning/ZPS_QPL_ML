echo '1st'
python main.py -lr 1e-4 -en 30 -bs 3000 -pf 3
echo '2nd'
python main.py -lr 1e-4 -en 30 -bs 1024 -pf 3
echo '3rd'
python main.py -lr 1e-4 -en 30 -bs 2048 -pf 3
echo '4th'
python main.py -lr 1e-4 -en 30 -bs 4096 -pf 3