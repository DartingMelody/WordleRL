##files used for the assignment of ranks to each word in wordspace/smallset. 
##For assigning ranks to smallset change wordspace to smallset at all locations in this files. 
with open('wordspace.txt', 'r') as f:
	Lines = f.readlines()

with open('freq_words.txt', 'r') as f:
	Linesf = f.readlines()

cnt = 1
comLines = {}
for line1 in Lines:
    # print ("line is "+line)
    for line in Linesf:
        # print(word)
        if (line == line1):
            # print ("line1 is "+line1)
            with open('wordspace1.txt', 'a') as the_file:
                linex = line1[:-1] + " "+str(cnt) + "\n"
                the_file.write(linex)
                isThere = True
                comLines[line1] = 1
                break
    cnt = cnt + 1

for line1 in Lines:
    print(line1)
    if(line1 not in comLines):
        with open('wordspace1.txt', 'a') as the_file:
            print(line1)
            linex = line1[:-1] + " "+str(cnt) + "\n"
            the_file.write(linex)
