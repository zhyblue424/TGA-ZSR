# TGA-ZSR
python -u ./TGA-ZSR.py

# TGA-ZSR-visual
python -u ./TGA-ZSR.py --atten_methods 'visual'

# CLIP
python -u ./TGA-ZSR.py --Method CLIP --Alpha 0.0 --Beta 0.0

# FT-Clean
python -u ./FT.py --Method FT-Clean --VPbaseline True

# FT-Adv.
python -u ./FT.py --Method FT-Standard --VPbaseline False

# TeCoA
python -u ./TGA-ZSR.py --Method TeCoA --Alpha 0.0 --Beta 0.0 --learning_rate 1e-5
