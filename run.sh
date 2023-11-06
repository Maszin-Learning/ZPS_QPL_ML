echo '1st'
python main.py -lr 1e-4 -en 50 -bs 64 -pf 3 -ds 10000 -g
echo '2nd'
python main.py -lr 1e-4 -en 50 -bs 128 -pf 3
echo '3rd'
python main.py -lr 1e-4 -en 50 -bs 256 -pf 3
echo '4th'
python main.py -lr 1e-4 -en 50 -bs 512 -pf 3