                 ELECTION ALG & DEADLOCK


1.CENTRALIZED

from collections import defaultdict

class graph:
    def __init__(self):
        self.graph = defaultdict(list)
    def addEdge(self, u, v):
        self.graph[u].append(v)
    def printgraph(self):
        for i in self.graph:
            print(i, end=" ")
            for j in self.graph[i]:
                print("->", j, end=" ")
            print()

s1 = graph()
s1.addEdge(3, 2)
s1.addEdge(2,1)

s2 = graph()
s2.addEdge(1, 2)

# central graph that combines s1 and s2
def central_graph(s1, s2):
    s3 = graph()
    for i in s1.graph:
        for j in s1.graph[i]:
            s3.addEdge(i, j)
    for i in s2.graph:
        for j in s2.graph[i]:
            s3.addEdge(i, j)
    return s3

s3 = central_graph(s1, s2)
# print(s3.graph[1])
# print(s3.graph)
#find cycle in s3

s1.printgraph()
print()
s2.printgraph()
print()
s3.printgraph()

def dfs(s3, v, visited):
    visited.add(v)
    for i in s3.graph[v]:
        # print(i)
        if i not in visited:
            if dfs(s3, i, visited):
                return True
        else:
            return True
    return False

def find_cycle(s3):
    visited = set()
    for i in s3.graph:
        if i not in visited:
            if dfs(s3, i, visited):
                return True
    return False


if find_cycle(s3):
    print("Deadlock found")
else:
    print("No deadlock")

def printgraph(s1):
    #print graph in a graphical way
    for i in s1.graph:
        print(i, end=" ")
        for j in s1.graph[i]:
            print("->", j, end=" ")
        print()












2. HIERARCHY

# python program to demonstrate the use of heirachial deadlock detection
from collections import defaultdict


class graph:
    def __init__(self):
        self.graph = defaultdict(list)

    def addEdge(self, u, v):
        self.graph[u].append(v)

    def printgraph(self):
        for i in self.graph:
            print(i, end="->")
            for j in self.graph[i]:
                print(",", j, end=" ")
            print()


n_sites = int(input("Enter the number of sites: "))
site_list = [graph()] * n_sites

# input from user

for i in range(n_sites):
    n = int(input("Enter the number of edges for site {}: ".format(i+1)))
    for j in range(n):
        u, v = map(int, input("Enter the edge (a->b) :").split())
        site_list[i].addEdge(u, v)

# site_list[0].printgraph()


def combine(s1, s2):
    s3 = graph()
    for i in s1.graph:
        for j in s1.graph[i]:
            s3.addEdge(i, j)
    for i in s2.graph:
        for j in s2.graph[i]:
            s3.addEdge(i, j)
    return s3


def dfs(s3, v, visited):
    visited.add(v)
    for i in s3.graph[v]:
        if i not in visited:
            if dfs(s3, i, visited):
                return True
        else:
            return True
    return False

def find_cycle(s3):
    visited = set()
    for i in s3.graph:
        if i not in visited:
            if dfs(s3, i, visited):
                return True
    return False

def detect_deadlock(site_list):
    while len(site_list) > 1:
        local_cordinator = []
        for i in range(0, len(site_list), 2):
            if i+1 < len(site_list):
                local_cordinator.append(combine(site_list[i], site_list[i+1]))
                lc = local_cordinator[-1]
                print(f"Local Coordinator of {i}, {i+1}")
                lc.printgraph()
                if find_cycle(lc):
                    print(f"Deadlock detected at coordinator")
                    lc.printgraph()
                    return
            else:
                local_cordinator.append(site_list[i])
                print("Odd number of sites so this is the last site")
                site_list[i].printgraph()
        site_list = local_cordinator
    print("No deadlock detected")


detect_deadlock(site_list)



3. PROBE

from collections import defaultdict
class sl:
    def __init__(self):
        self.gg=defaultdict(list)
    def inh(self,u,v,n):
        n[u].append(v)
    def merge1(self,ss1,ss2):
        ss2=self.gg.copy()
        self.gg.clear()
        for f in(ss1,ss2):
            for k,v in f.items():
                for p in v:
                    self.gg[k].append(p)
    def check(self,di,k):
        for i in di.keys():
            if i==k:
                return 1
        return 0
    def sol(self,k,j):
        ls=[]
        lss=[]
        m={}
        ls.append(k)
        lss.append(k)
        while ls:
            t=ls.pop(0)
            for i in j[t]:
                if i not in m.keys():
                    m[i]=[list((k,t,i))]
                else:
                    m[i].append(list((k,t,i)))
                # print(m)
                if i not in lss:
                    lss.append(i)
                    ls.append(i)
        return m

s=sl()
dh=defaultdict(list)
dr=defaultdict(list)
r=int(input("Enter the number of resources : "))
p=int(input("Enter the number of processes : "))
lss=[]
while(True):
    r1=int(input("Enter the resource : "))
    if r1<0:
        break
    if r1==0 or r1>r:
        print("Enter valid resource")
    q = int(input("Enter the process holding resources : "))
    if q==0 or q>p:
        print("Enter valid process")
    qq = int(input("Enter the process requesting resources : "))
    if qq==0 or qq>p:
        print("Enter valid process")
    w=str(input("Enter the site for the processes : "))
    if w not in lss:
        lss.append(w)
        vars()[w]=defaultdict(list)
    s.inh(q,qq,vars()[w])
    dh[r1].append(q)
    dr[r1].append(qq)
for i in lss:
    print(f"Site {i}:\n{dict(vars()[i])}")
flag1=1
y1=int(input("Enter requesting process: "))
x1=int(input("Enter resource holding process: "))
flag=1
for i in lss:
    if flag==1:
        s.gg=vars()[i].copy()
        flag=0
    else:
        s.merge1(vars()[i],s.gg)
print(f"WFG:\n{dict(s.gg)}")
d1=s.sol(y1,s.gg)
print(f"Probe message received:\n{dict(d1)}")
if(s.check(d1,y1)==1):
    print("Deadlock")
else:
    print("No deadlock")



4. BULLY

class process:
    def __init__(self, id, priority,active=True):
        self.id = id
        self.priority = priority
        self.active = active

def bully():
    n = int(input("Enter number of processes: "))
    processes = []
    for i in range(n):
        id = int(input("Enter process id: "))
        priority = int(input("Enter process priority: "))
        active = int(input("Enter process status (0 or 1): "))
        processes.append(process(id, priority,active))

    processes.sort(key=lambda x : x.priority, reverse=True)

    while True:
        start = int(input("Enter starting process: "))
        print("Election Begins")

        if start not in [x.id for x in processes]:
            print("Invalid starting process")
            break

        else:
            arr=[start]
            while len(arr)!=0:
                flag=0
                start = arr.pop(-1)
                for i in range(len(processes)):
                    if processes[i].id == start:
                        index=i
                for i in range(index-1, -1, -1):
                    if processes[i].active:
                        print("--> Message is sent from process "+str(start)+" to process "+str(processes[i].id))
                        arr.append(processes[i].id)
                        print("<-- Reply from process "+str(processes[i].id))
                        flag = 1
                    else:
                        print("--> Message is sent from process "+str(start)+" to process "+str(processes[i].id))
                        print("<-- No Reply from process "+str(processes[i].id))
                if flag==0:
                    coordinator = start
                    break

                arr.sort(reverse=True)
            
            print("Election Ends")
            print("Coordinator is process "+str(coordinator))

            print("Do you want to continue? (y/n)")
            status = input()
            if status=='n':
                break
            else:
                print("Enter the process to crash: ")
                crash = int(input())
                for i in range(len(processes)):
                    if processes[i].id == crash:
                        processes[i].active = False
                        break

bully()





5. RING

class process:
    def __init__(self, id, priority,active=True):
        self.id = id
        self.priority = priority
        self.active = active

def ring():
    n = int(input("Enter number of processes: "))
    processes = []
    for i in range(n):
        id = int(input("Enter process id: "))
        priority = int(input("Enter process priority: "))
        active = int(input("Enter process status (0 or 1): "))
        processes.append(process(id, priority,active))

    # processes.sort(key=lambda x : x.priority, reverse=True)
    while True:
        ring_arr=[]
        start = int(input("Enter starting process: "))
        for i in range(len(processes)):
            if processes[i].id == start:
                index=i
                break
        next = (index+1)%n
        while processes[index].id!=start or len(ring_arr)==0:
            if processes[index].active and processes[next].active:  
                print("--> Message is sent from process "+str(processes[index].id)+" to process "+str(processes[next].id))
                ring_arr.append(processes[index])
                # print(ring_arr[-1].id)
                index=next
                next = (next+1)%n
            elif processes[index].active and not processes[next].active:
                print("--> Message is sent from process "+str(processes[index].id)+" to process "+str(processes[next].id))
                print("<-- No Reply from process "+str(processes[next].id))
                next = (next+1)%n

        print("Election Ends")
        ring_arr.sort(key=lambda x : x.priority, reverse=True)
        print("Coordinator is process "+str(ring_arr[0].id))

        print("Do you want to continue? (y/n) : ")
        status = input()
        if status=='n':
            break
        else:
            print("Enter the process to crash: ")
            crash = int(input())
            for i in range(len(processes)):
                if processes[i].id == crash:
                    processes[i].active = False
                    break
ring()


6. MPI Matrix Multiplication using Point to Point

#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <iostream>
using namespace std;
#define N 4
int main(int argc, char **argv) {
int rank, size;
int a[N][N], b[N][N], c[N][N];
int i, j, k, sum;
MPI_Init(&argc, &argv);
MPI_Comm_rank(MPI_COMM_WORLD, &rank);
MPI_Comm_size(MPI_COMM_WORLD, &size);
if (size != 2) {
fprintf(stderr, "This program requires exactly 2 processes.\n");
MPI_Abort(MPI_COMM_WORLD, 1);
}
for (i = 0; i < N; i++) {
for (j = 0; j < N; j++) {
a[i][j] = i+j;
b[i][j] = i-j;
c[i][j] = 0;
}
}
if(rank==0)
{
cout<<"Matrix A: "<<endl;
for(int i=0;i<N;i++)
{
cout<<endl;
for (int j=0;j<N;j++)
{
cout<<a[i][j]<<" ";
}
}
cout<<endl<<endl;
cout<<"Matrix B: "<<endl;
for(int i=0;i<N;i++)
{
cout<<endl;
for (int j=0;j<N;j++)
{
cout<<b[i][j]<<" ";
}
}
cout<<endl<<endl;
}

// scatter matrix A to process 0 and B to process 1

MPI_Scatter(a, N*N/2, MPI_INT, a, N*N/2, MPI_INT, 0, MPI_COMM_WORLD);
MPI_Scatter(b, N*N/2, MPI_INT, b, N*N/2, MPI_INT, 1, MPI_COMM_WORLD);

// compute matrix multiplication
for (i = 0; i < N/2; i++) {
for (j = 0; j < N; j++) {
sum = 0;
for (k = 0; k < N; k++) {
sum += a[i][k] * b[k][j];
}
c[i][j] = sum;
}
}





// gather results from both processes

MPI_Gather(c, N*N/2, MPI_INT, c, N*N/2, MPI_INT, 0, MPI_COMM_WORLD);

// print result on process 0

if (rank == 0) {
printf("Result of Matrix Multiplication :\n");
for (i = 0; i < N; i++) {
for (j = 0; j < N; j++) {
printf("%d ", c[i][j]);
}
printf("\n");
}
}
MPI_Finalize();
return 0;
}







7. RPC 

a)Amstrong

#include <stdio.h>
#include <math.h>

int isArmstrong(int num) {
    int sum = 0, temp = num, digit;
    int order = (int)log10(num) + 1;

    while (temp != 0) {
        digit = temp % 10;
        sum += pow(digit, order);
        temp /= 10;
    }

    if (sum == num) {
        return 1;
    } else {
        return 0;
    }
}

int main() {
    int n, count = 0, i = 1, sum = 0;

    printf("Enter the value of n: ");
    scanf("%d", &n);

    while (count < n) {
        if (isArmstrong(i)) {
            sum += i;
            count++;
        }
        i++;
    }

    printf("The sum of the first %d Armstrong numbers is %d", n, sum);

    return 0;
}



b)Krishnamoorthy series

#include <stdio.h>
#include <math.h>

int isKrishnamoorthy(int num) {
    int sum = 0, prod = 1, temp = num, digit;

    while (temp != 0) {
        digit = temp % 10;
        sum += digit;
        prod *= digit;
        temp /= 10;
    }

    if (sum * prod == num) {
        return 1;
    } else {
        return 0;
    }
}

int main() {
    int n, count = 0, i = 1;

    printf("Enter the value of n: ");
    scanf("%d", &n);

    printf("Krishnamoorthy numbers up to %d:\n", n);

    while (i <= n) {
        if (isKrishnamoorthy(i)) {
            printf("%d\n", i);
            count++;
        }
        i++;
    }

    if (count == 0) {
        printf("No Krishnamoorthy numbers found up to %d\n", n);
    }

    return 0;
}



c) fibonacci_series

#include <stdio.h>

int main() {
   int n, i, t1 = 0, t2 = 1, nextTerm;

   printf("Enter the number of terms: ");
   scanf("%d", &n);

   printf("Fibonacci Series: ");

   for (i = 1; i <= n; ++i) {
      printf("%d, ", t1);
      nextTerm = t1 + t2;
      t1 = t2;
      t2 = nextTerm;
   }

   return 0;
}


d) sine series

#include <stdio.h>
#include <math.h>

int main() {
   double x, sinx = 0.0;
   int i, j, n;

   printf("Enter the value of x (in degrees): ");
   scanf("%lf", &x);

   // Convert degrees to radians
   x = x * M_PI / 180.0;

   printf("Enter the number of terms: ");
   scanf("%d", &n);

   // Calculate the sine series using the Taylor series expansion
   for (i = 1, j = 1; i <= n; i++, j += 2) {
      if (i % 2 != 0) {
         sinx += pow(x, j) / factorial(j);
      } else {
         sinx -= pow(x, j) / factorial(j);
      }
   }

   printf("sin(%lf) = %lf\n", x, sinx);

   return 0;
}

// Function to calculate the factorial of a number
int factorial(int n) {
   if (n == 0 || n == 1) {
      return 1;
   } else {
      return n * factorial(n - 1);
   }
}


e) cosine series

#include <stdio.h>
#include <math.h>

int main() {
   double x, cosx = 1.0;
   int i, j, n;

   printf("Enter the value of x (in degrees): ");
   scanf("%lf", &x);

   // Convert degrees to radians
   x = x * M_PI / 180.0;

   printf("Enter the number of terms: ");
   scanf("%d", &n);

   // Calculate the cosine series using the Taylor series expansion
   for (i = 1, j = 2; i <= n; i++, j += 2) {
      if (i % 2 != 0) {
         cosx -= pow(x, j) / factorial(j);
      } else {
         cosx += pow(x, j) / factorial(j);
      }
   }

   printf("cos(%lf) = %lf\n", x, cosx);

   return 0;
}

// Function to calculate the factorial of a number
int factorial(int n) {
   if (n == 0 || n == 1) {
      return 1;
   } else {
      return n * factorial(n - 1);
   }
}


f) binary to octal

#include <stdio.h>
#include <math.h>

int binaryToDecimal(int binary)
{
    int decimal = 0, i = 0;
    while (binary != 0)
    {
        int remainder = binary % 10;
        binary /= 10;
        decimal += remainder * pow(2, i);
        i++;
    }
    return decimal;
}

int main()
{
    int binary, decimal = 0, octal = 0, i = 0;
    printf("Enter a binary number: ");
    scanf("%d", &binary);

    decimal = binaryToDecimal(binary);

    while (decimal != 0)
    {
        octal += (decimal % 8) * pow(10, i);
        decimal /= 8;
        i++;
    }

    printf("Octal number: %d", octal);

    return 0;
}


g) octal to binary

#include <stdio.h>

int main()
{
    int octal, decimal = 0, binary = 0, i = 0;

    printf("Enter an octal number: ");
    scanf("%d", &octal);

    // Convert octal to decimal
    while (octal != 0)
    {
        decimal += (octal % 10) * pow(8, i);
        i++;
        octal /= 10;
    }

    i = 1;
    // Convert decimal to binary
    while (decimal != 0)
    {
        int remainder = decimal % 2;
        binary += remainder * i;
        i *= 10;
        decimal /= 2;
    }

    printf("Binary number: %d", binary);

    return 0;
}


