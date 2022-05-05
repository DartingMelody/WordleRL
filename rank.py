with open('/Users/tarannumkhan/Desktop/WordleRL/wordspace.txt', 'r') as f:
	Lines = f.readlines()

with open('/Users/tarannumkhan/Desktop/WordleRL/freq_words.txt', 'r') as f:
	Linesf = f.readlines()

cnt = 1
comLines = {}
for line1 in Lines:
    # print ("line is "+line)
    for line in Linesf:
        # print(word)
        if (line == line1):
            # print ("line1 is "+line1)
            with open('/Users/tarannumkhan/Desktop/WordleRL/wordspace1.txt', 'a') as the_file:
                linex = line1[:-1] + " "+str(cnt) + "\n"
                the_file.write(linex)
                isThere = True
                comLines[line1] = 1
                break
    cnt = cnt + 1

for line1 in Lines:
    print(line1)
    if(line1 not in comLines):
        with open('/Users/tarannumkhan/Desktop/WordleRL/wordspace1.txt', 'a') as the_file:
            print(line1)
            linex = line1[:-1] + " "+str(cnt) + "\n"
            the_file.write(linex)
