import bittensor as bt
import random
import string


if __name__ == "__main__":
	subtensor = bt.subtensor(network='finney')
		# while True:
	for _ in range(10000):
		try:
			# string with 6 random letters and numbers
			wallet_name = ''.join(random.choices(string.ascii_letters + string.digits, k=6))
			wallet = bt.wallet(name="finney_contest", hotkey=wallet_name).create(coldkey_use_password=False).register(subtensor=subtensor, netuid=1)

			if wallet.is_registered:
				print('registered')
				continue
		except:
			print('error')
			continue
