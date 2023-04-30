#include <bits/stdc++.h>
#include <iostream>
#include <stdlib.h>
#include <string>

using namespace std;

class btree {
public:
    int data;
    btree* next;
};

// This function prints contents of linked list
// starting from the given node
void printList(btree* n)
{
    while (n != NULL) {     // as long as we are pointing to a node and not NULL, then we will print its contents
        cout << n->data << endl;
        n = n->next;
    }
}

// Driver code
int main()
{
    // initialization
    btree* first = NULL; // head of the list
    btree* second = NULL;
    btree* third = NULL;

    // dynamically allocate 3 nodes in the heap
    first = new btree();
    second = new btree();
    third = new btree();

    first->data = 1; // assign data in first node
    first->next = second; // Link first node with second

    second->data = 2; // assign data to second node
    second->next = third;

    third->data = 3; // assign data to third node
    third->next = NULL;

    printList(first);

    return 0;
}

// References
// Parts of this code (not all) is from https://www.geeksforgeeks.org/linked-list-set-1-introduction/
// Alex Hu and Shaun Gunawardane also helped me to write this code.