echo '1st'
python main.py -lr 5e-5 -en 30 -bs 64 -pf 3 -ds 20000 -g
echo '2nd'
python main.py -lr 5e-5 -en 30 -bs 128 -pf 3
echo '3rd'
python main.py -lr 5e-5 -en 30 -bs 256 -pf 3
echo '4th'
python main.py -lr 5e-5 -en 30 -bs 512 -pf 3