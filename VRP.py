import pandas as pd
import pulp
from sklearn.metrics.pairwise import manhattan_distances
import matplotlib.pyplot as plt

locations = pd.read_csv('locations.csv')
locations_list = locations['Location'].unique()
n_locations = len(locations_list)
destinations_list = locations_list[1:]
locations = locations.set_index(['Location'])
capacity = 6

distances = pd.DataFrame(manhattan_distances(locations.values, locations.values, sum_over_features=True))

vehicles = [1,2,3]
n_vehicles = len(vehicles)

#Definition of the variables
x = pulp.LpVariable.dicts("x",
                          ((i,j,k) for i in locations_list for j in locations_list for k in vehicles),
                          cat='Binary')

u = pulp.LpVariable.dicts("u",((i) for i in locations_list),
                          lowBound=0)

#Type and name of the problem

model = pulp.LpProblem("Vehicle routing problem", pulp.LpMinimize)

#Definition of the Objective Function
model += pulp.lpSum([x[(i,j,k)]*distances.loc[i,j] for i in locations_list for j in locations_list for k in vehicles])

#Definition of constraints

#1 - #2 destinations are visited only once

for i in destinations_list:
    model += pulp.lpSum(x[(i,j,k)] for j in locations_list for k in vehicles) == 1

for j in destinations_list:
    model += pulp.lpSum(x[(i, j, k)] for i in locations_list for k in vehicles) == 1

#3 the number of vehicles that leave the origin is the same than the number that arrives

model += pulp.lpSum(x[(0,j,k)] for j in locations_list for k in vehicles) == n_vehicles

model += pulp.lpSum(x[(i,0,k)] for i in locations_list for k in vehicles) == n_vehicles

#4 no subtours allowed

for k in vehicles:
    for i in locations_list:
        for j in destinations_list:
            model += u[i]-u[j]+n_locations*x[(i,j,k)] <= n_locations-1


#each destination is visited only once

for k in vehicles:
    for j in destinations_list:
        model += pulp.lpSum(x[(i,j,k)] for i in locations_list) == pulp.lpSum(x[(j,i,k)] for i in locations_list)


for k in vehicles:
    model += pulp.lpSum(x[(i,j,k)] for i in locations_list for j in destinations_list) <= capacity


#solve the model
#model.solve(pulp.GLPK_CMD(options=['--mipgap', '0.01']))


#model.solve(solver=pulp.GUROBI(MIPGap = 0.01))


model.solve(pulp.PULP_CBC_CMD(maxSeconds=60,
                              msg=1,
                              fracGap=0.1))

model.solve(pulp.GUROBI(#TimeLimit=60,
                        msg=1,
                        mip=1,
                        MIPGap =0.1))
# print solution status
print(pulp.LpStatus[model.status])

# Print our objective function value (Total Distance)
print("The value of the objective function is: "+ str(pulp.value(model.objective)))

# output=[]
#
# for v in model.variables():
#     var_output= {
#         'variable':v.name ,
#         'status':v.varValue
#     }
#
#     output.append(var_output)
#
#
# output_df = pd.DataFrame.from_records(output)
#
# active_variables = output_df[output_df['status']>0]
# active_variables['type']=active_variables['variable'].str[0]
# active_variables = active_variables[active_variables['type'] == 'x']
#
# print(active_variables)


output_x=[]

for i,j,k in x:
    x_output = {
        'vehicle':k,
        'start': i,
        'end':j,
        'status':x[(i,j,k)].varValue
    }
    output_x.append(x_output)

x_output_df = pd.DataFrame.from_records(output_x)

x_output_df = x_output_df[['vehicle','start','end','status']]
x_output_df = x_output_df.set_index(['vehicle','start','end'])
x_output_df = x_output_df.sort_index()

x_active_output = x_output_df[x_output_df['status'] == 1]

print(x_active_output)

x_active_origins = x_active_output.reset_index()
x_active_origins = x_active_origins.set_index(['vehicle','start'])


routes={}


for k in vehicles:
    origin = 0
    route = []
    route.append(origin)
    while True:
        destination = int(x_active_origins.loc[(k, origin), 'end'])
        route.append(destination)
        origin = destination
        if destination == 0:
            break
    routes[k] = route



for k in vehicles:
    selected_route = locations.loc[routes[k],:]
    x_values = list(selected_route['X Axis'])
    y_values = list(selected_route['Y Axis'])
    plt.plot(x_values,y_values,'.',linestyle='dashed')

x_coordinates = list(locations['X Axis'])
y_coordinates = list(locations['Y Axis'])

for i in locations_list:
    label = locations_list[i]
    coord = (x_coordinates[i],y_coordinates[i])

    plt.annotate(label,
                 coord,
                 textcoords="offset points",
                 xytext=(0,10),
                 ha='center')

plt.grid(b=None,which='both',axis='both')
plt.legend(vehicles, loc='upper left')

plt.show()


