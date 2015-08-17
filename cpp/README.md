### Cpp Reference

#### STL

1. when checking whether a container is empty, use '''empty()''' function, not '''size() == 0'''
2. The effective way to remove element in container: '''c.erase(remove(c.begin(), c.end(), ele),c.end())'''
3. use reserve() to allocate memory and wait for use.
4. "swap trick": vector<T>(origin).swap(origin);  string(s).swap(s)
5. In associate container, it use equivalance(not equality) to compare elements.
6. When using "[]" to insert data, firstly it will allocate a new space, initialize the element, then assign a value.
7. Change const_iterator to iterator: '''advance(iter, distance(iter, ci))'''
8. Change reverse_iterator to iterator: v.insert(ri.base(), val);   v.erase((++ri).base())
9. Sort STL Algorithm: partition, stable_partition, nth_element, partial_sort, sort, stable_sort
10. Use '''accumulate''' and '''for_each''' to clean your code

