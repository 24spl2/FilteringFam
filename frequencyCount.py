class frequencyCount:


    #notes: first make a dict/hashmap to decide the most important 50 or so words. 
    #Then make a second hashmap to determine with an array importance
    # The food is hot the 
    # [ 0 2 0 1 0 0] etc. etc.
    
    #useful things
    '''
    split(at a character that you want to split at)
    '''

    if __name__ == '__main__':

        #BEGIN make frequency
        f = open("_chat_text_only.txt", "r")
        text = f.read().lower()
        #will split all the text on the space into a list
        total_words = text.split()

        frequency = {}
        for word in total_words:
            if word in frequency.keys():
                frequency[word] += 1
            else:
                frequency[word] = 1

        final_frequency = {}
        for word in frequency:
            if frequency[word] > 18:
                final_frequency[word] = frequency[word]

        frequency = final_frequency
        #print(sorted(frequency.items(), key=lambda item: item[1], reverse = True))
        #END make frequency count

        #BEGIN make thetas
        tracker = frequency
        #zeros out values
        for theta in tracker:
            tracker[theta] = 0
        print(tracker)
        print(len(tracker))
        writeWeights = open("thetas.txt", "a")

        #Initialize column
        writeLine = ""
        for theta in tracker:
            writeLine += theta + ", "
        
        writeLine += "\n"
        writeWeights.write(writeLine)

        s = open("_chat_text_only.txt", "r")
        #1st loop: goes through entire text file
        for line in s:
            #READING
            lineWords = line.split()
            #2nd for loop: looks at each word in text
            for singleWord in lineWords:
                #updates the dict
                if singleWord in tracker.keys():
                    tracker[singleWord] += 1

            #WRITING
            #formats each theta to be comma separated
            for theta in tracker:
                writeLine += str(tracker[theta]) + ", "
            #breaks to next line
            writeLine += "\n"
            writeWeights.write(writeLine)

            #RESET
            writeLine = ""
            for theta in tracker:
                tracker[theta] = 0
            
        
        #END make thetas