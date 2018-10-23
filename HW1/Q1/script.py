import argparse
import requests
import csv
import time


#Take args from cmd line
parser = argparse.ArgumentParser(description='TMDB database download')
parser.add_argument('API_key', metavar='Key', type=str, help='your TMBD API key')
args = parser.parse_args()



# Get the list of popular movies
print("Plz be patient, maybe go get a cup of coffee, this gonna take some time...")
page = 1
count = 0
with open('movie_id_name.csv', 'w') as csvfile:
	while count < 300:
		r = requests.get('https://api.themoviedb.org/3/discover/movie?api_key=' + args.API_key +
                         "&language=en-US&sort_by=popularity.desc&include_adult=false&include_video=false&page=" +
                         str(page) + "&primary_release_date.gte=2000&with_genres=35")
		for item in r.json()['results']:
			count += 1
			moviewriter = csv.writer(csvfile)
			moviewriter.writerow([item['id'],item['original_title']])
			if count == 300:
				break
		page +=1



# Get similar movies
print("Hold on. I'm looking for the similar movies.")
with open('movie_id_name.csv', 'r') as csvfile:
	reader = csv.reader(csvfile, delimiter=",")
	sim = []
	for line in reader:
		movieId = line[0]
		r = requests.get('https://api.themoviedb.org/3/movie/' + str(
			movieId) + '/similar?api_key=' + args.API_key + '&language=en-US&page=1')
		time.sleep(0.3)
		if r.json()['total_results'] >= 5:
			count_2 = 5
		else:
			count_2 = r.json()['total_results']
		for indexNum in range(count_2):
			sim.append([str(movieId), str(r.json()['results'][indexNum]['id'])])


# Remove duplicate files
print("Gonna remove the duplicate files. Just a minute.")
rlist = []
for item1 in sim:
	for item2 in sim:
		if item1 == [item2[1], item2[0]]:
			if int(item1[0]) < int(item2[0]):
				rlist.append(sim.index(item2))
			else:
				rlist.append(sim.index(item1))

for j in sorted(rlist, reverse=True):
	del sim[j]

with open('movie_ID_sim_movie_ID.csv', 'w', newline='\n') as csvfileSim:
	writer = csv.writer(csvfileSim)
	writer.writerow(['Source', 'Target'])
	writer.writerows(sim)

print("I think we cool now, buddy.")
