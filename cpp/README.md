### Cpp Reference

#### STL

when checking whether a container is empty, use 
```c++
empty()
```
function, not 
```c++
size() == 0
```
---
The effective way to remove element in container: 
```c++
c.erase(remove(c.begin(), c.end(), ele),c.end())
```
use 
```c++
reserve() 
```

to allocate memory and wait for use.
---
"swap trick": 
```c++
vector<T>(origin).swap(origin);  
string(s).swap(s)
```
---
In associate container, it use equivalance(not equality) to compare elements.
```c++
if(a < b){
  return -1;
}else if(b < a){
  return 1;
}else{
  return 0;//may be they still not the same, but equivalance.
```
---
When using "[]" to insert data, firstly it will allocate a new space, initialize the element, then assign a value.
```c++
m[1] = 1.50;

equals like:

pair<int, float> result = m.insert(make_pair<int, float>(1, 0));
result.second = 1.50;
```
---
Change const_iterator to iterator: 
```c++
advance(iter, distance(iter, ci))
```
---
Change reverse_iterator to iterator: 
```c++
v.insert(ri.base(), val);   v.erase((++ri).base())
```
---
Sort STL Algorithm: 
```c++
partition(v.begin(), v.end(), [](T ele){return judge(ele);}); 
stable_partition(v.begin(), v.end(), [](T ele){return judge(ele);});
nth_element(v.begin(), v.begin() + nelement, v.end(), [](T ele1, T ele2){compare(ele1, ele2)});
partial_sort(v.begin(), v.begin() + nelemnt, v.end(), [](T ele1, T ele2){compare(ele1, ele2)});
sort(v.begin(), v.end(), [](T ele1, T ele2){compare(ele1, ele2)}); 
stable_sort(v.begin(), v.end(), [](T ele1, T ele2){compare(ele1, ele2)}); 
```
---
Use accumulate and for_each to clean your code
```c++
res = accumulate(v.begin(), v.end(), initial_val);
for_each(v.begin(), v.end(), func);
```

