import os


PATH = os.path.dirname(os.path.realpath(__file__))

# fashion_splt = {} 

# fashion_str = { 'background': """ Cufflinks, Rompers, Laptop Bag, Sports Sandals, Hair Colour, 
# Suspenders, Trousers, Kajal and Eyeliner, Compact, Concealer, Jackets, Mufflers,
# Backpacks, Sandals, Shorts, Waistcoat, Watches, Pendant, Basketballs, Bath Robe,
# Boxers, Deodorant, Rain Jacket, Necklace and Chains, Ring, Formal Shoes, Nail Polish,
# Baby Dolls, Lip Liner, Bangle, Tshirts, Flats, Stockings, Skirts, Mobile Pouch, Capris,
# Dupatta, Lip Gloss, Patiala, Handbags, Leggings, Ties, Flip Flops, Rucksacks, Jeggings,
# Nightdress, Waist Pouch, Tops, Dresses, Water Bottle, Camisoles, Heels, Gloves, Duffel Bag, 
# Swimwear, Booties, Kurtis, Belts, Accessory Gift Set, Bra """, 'evaluation' : """Jeans, Bracelet, Eyeshadow, Sweaters, Sarees, Earrings, Casual Shoes,
# Tracksuits, Clutches, Socks, Innerwear Vests, Night suits, Salwar, Stoles, Face Moisturisers, 
# Perfume and Body Mist, Lounge Shorts, Scarves, Briefs, Jumpsuit, Wallets,
# Foundation and Primer, Sports Shoes, Highlighter and Blush, Sunscreen, Shoe Accessories, 
# Track Pants, Fragrance Gift Set, Shirts, Sweatshirts, Mask and Peel,
# Jewellery Set, Face Wash and Cleanser, Messenger Bag, Free Gifts, Kurtas, Mascara,
# Lounge Pants, Caps, Lip Care, Trunk, Tunics, Kurta Sets, Sunglasses, Lipstick, Churidar,
# Travel Accessory"""}

# for key_, str_ in fashion_str.items():
#     fashion_splt[key_] = str_
#     fashion_splt[key_] = fashion_splt[key_].split(',')
#     fashion_splt[key_] = [i.strip() for i in fashion_splt[key_]]

# # import pdb; pdb.set_trace()




DATA_PATH = '/home/ubuntu/pavanteja/few-shot/data'
DATA_PATH_Fashion = '/mnt/datasets/datasets/fashion-dataset/fashion-dataset/images_resized/'

EPSILON = 1e-8

if DATA_PATH is None:
    raise Exception('Configure your data folder location in config.py before continuing!')
