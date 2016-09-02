import sys
import re
import editdistance

# open locations file, which is first command line argument
# NOTE: sys.argv[0] is the Python file itself, so we skip it
locations_file = open(sys.argv[1], 'r')
# open tweet file, which is second command line argument
tweets_file = open(sys.argv[2], 'r')

# put data into arrays
locations = locations_file.readlines()
tweets = tweets_file.readlines()

# close files
locations_file.close()
tweets_file.close()

# analyze each tweet
for t in tweets :
	tokens = t.split('\t')
	# tweet text is in tokens[2]
	if len(tokens) == 3 :
		print("--------------------------------------------------------------")
		print (tokens[2] + '\n')
		words = tokens[2].split(' ')
		toks = []
		# create a list of tokens that do not contain special characters
		for w in words :
			# ignore words that have special characters, because they are likely not location names
			# use regular expressions to check for special characters, ignore word if it contains special characters
			first = w[:-1]
			last = w[-1:]
			# if the string ends in punctuation but has no other special characters, cut off punctuation and then try without punctuation
			if bool(re.search('\W', last)) :
				if len(first) > 0 :
					toks.append(first)
			# only allow strings without special characters, since locations will likely not have special characters
			# additionally, don't allow empty strings for obvious reasons
			if not bool(re.search('\W', w)) and not bool(re.search('\d', w)) and len(w) > 0 :
				# word does not contain special characters, add to list of tokens
				toks.append(w)
			# use global edit distance to find closest match
			# accomodate multi-word locations by comparing chunks of the tweet with the same number of tokens
		for loc in locations :
			# every location ends in a weird blank character that I can't get rid of
			loc = loc[:-1]
			# ignore locations with numbers and special characters
			if loc is not '\n' and loc is not '' and not bool(re.search('\d', loc)) :
				num_tokens = len(loc.split(' '))
				#print(num_tokens)
				# keep front and back pointers
				front = 0
				back = num_tokens
				# ignore location if tweet isn't long enough to accomodate multi-word location
				while back <= len(toks) :
					q = ""
					for i in range(front, back) :
						q += toks[i] + ' '
					q = q[:-1]

					ed = editdistance.eval(q, loc)

					# short term so edit distance should be close to be considered a misspelled location
					# allow a greater global edit distance for longer terms
					min_term_length = min(len(q), len(loc))
					if min_term_length <= 5 :
						if ed == 0 :
							print("Assumed location: " + q)
							print("Actual location: " + loc)
							print("Edit distance: " + str(ed))
							print('\n')
					elif min_term_length > 5 and min_term_length <= 10 :
						if ed <= 1 :
							print("Assumed location: " + q)
							print("Actual location: " + loc)
							print("Edit distance: " + str(ed))
							print('\n')
					elif min_term_length > 10 and min_term_length <= 15 :
						if ed <= 2 :
							print("Assumed location: " + q)
							print("Actual location: " + loc)
							print("Edit distance: " + str(ed))
							print('\n')
					elif min_term_length > 15 :
						if ed <= 4 :
							print("Assumed location: " + q)
							print("Actual location: " + loc)
							print("Edit distance: " + str(ed))
							print('\n')

					front += 1
					back += 1

"""
This is the manual global edit distance that was implemented before using the Python package

grid = []
# initialize first row
grid.append(list(range(0, len(loc) + 1)))
# initialize first columns
for i in range(1, len(w) + 1) :
	grid.append([i])
	# complete global edit distance table
	# w is the rows and loc is the columns
	for i in range(1, len(w) + 1) :
 		for j in range(1, len(loc) + 1) :
			# eq is 0 if the letters being currently inspected are equal, otherwise 2
			eq = 0 if w[i-1].lower() == loc[j-1].lower() else 2
			# fill in table entry based on recurrence relation
			grid[i].append(min(grid[i-1][j] + 1, grid[i][j-1] + 1, grid[i-1][j-1] + eq))
"""
