driver类：
```cpp
using LocalQueueT = std::map<Stamp, WorkSet>
using WorkSet = std::vector<ItemT> // 一个revisit指针的数组
using ItemT = std::unique_ptr<Revisit>;

std::vector<Execution> execStack; // 一个execution的数组
std::unique_ptr<llvm::Interpreter> EE; // 一个interpreter的指针
std::shared_ptr<const Config> userConf; // 一个配置的指针

```
getGraph()是跳转了两层，先调用getExecution()，获取execStack尾部的execution，然后获取这个execution的成员变量graph
getworkqueue() 也是一层，先调用getExecution()，在获取它的workqueue成员
在execution里，有一个LocalQueueT workqueue;因为execution是定义在driver里的，所以用的driver的typedef。所以每个execution有一个graph和一个stamp到workset的map，且只有这两个成员。  

runAsMain函数定义在execution.cpp文件里，但是是Interpreter的成员函数。Interpreter里面有一个driver的裸指针。先初始化一些状态，然后调用driver的handleExecutionStart()方法，这个函数获取一个 (tid, 0) 的event，并且将每个函数的栈初始化。
然后调用interpreter的run()方法，这个run方法应该是llvm提供的，并没有在代码中，应该是用的crtp，只定义了一些处理读写等指令的函数。
结束之后调用driver的handleExecutionEnd()方法。

在explore函数的外层while循环中，先调用runasmain，这个函数执行中会往workqueue中加东西，而workqueue是execution中的成员，所以往当前execution中的workqueue中加 (stamp, revisit指针) 的pair。

进入内层循环中，也会从当前execution的workqueue中pop东西，pop出(stamp, revisit指针)，如果revisit是空指针，则pop出一个execution并继续，如果execution也空了，则直接返回。

在往worklist中push的时候，永远是往当前execution中的worklist中push，只是在backward revisit的时候，会拷贝一份graph并裁剪，然后构造一个新的executionpush到execution stack中。

在结束的时候，会往workqueue中push一个 (0, rerun) ，因为pop是从后往前，所以如果getNextItem() pop出来是 (0, rerun)则直接跳出内层循环重新runasmain了。getNextItem()会遍历map，map中是一个stamp对应一个vector，所以还需要调用getNext()方法获取这个vector中的下一个元素，而每次都从vector末尾取，而addtoworklist中也是push_back，所以后加入的会被先取出来。

也就是说，原来的estimation版本，外层循环只有在runasmain末尾才会push进一个rerun，进入内层循环之后，getNextItem()会得到一个rerun的ItemT，即一个revisit指针，在执行restrictAndRevisit会返回返回true，则跳出内层循环，重新runasmain

TODO：
调试revisit会不会调用 handleexecutionend函数，以及是否应该在push了revisit之后再pushrerun呢，这样会导致 (0, rerun, rerun...) 一堆rerun。注意RerunForwardRevisit就是在estimation mode独有的，只在handleend中使用的
--> revisit不会重新跑一遍，而是检查是否consistent，之后退出内层循环，在外层循环重新runasmain的时候从裁剪后的graph开始，而且只有runasmain函数会调用 handleExecutionEnd，虽然会从头开始运行，但是不会该prefix的部分

backward revisit的时候会拷贝一个graph，然后push到execution stack的数组里，然后changerf，changerf获取的execution是从尾部获取的，也就是新push进去的execution


## rf approximation: 

参数为一个read的label，首先调用getCoherentStores获取一组coherent的stores，如果不是CasReadLabel或FaiReadLabel则直接返回，否则从中删除一些：FaiReadLabel和CasReadLabel，并且符合一些附加条件的写。

getCoherentStores是各个内层模型自己定义的，例如在RC11中：

在GenMCDriver::constructBackwardRevisit(const ReadLabel* rLab, const WriteLabel* sLab) 里，如果config中没有helper，只有一行
```cpp
return std::make_unique<BackwardRevisit>(rLab, sLab, getRevisitView(rLab, sLab));

在getRevisitView函数中，先获取读操作的 PredsView，调用的是
getViewFromStamp()
这个函数从一个stamp获取，遍历每个线程，每个线程从后往前找，找到第一个stamp <= 输入的stamp的位置
返回的是每个线程在 rLab 之前（stamp之前）的位置

（getRevisitView）然后调用updatePredsWithPrefixView(g, *preds, getPrefixView(sLab)); 这个函数第一个参数是const，所以不会修改graph。第二个参数是从read获取的preds，即所有stamp小于read的位置，第三个参数是从 sLab获取的 prefixView，这个prefixView是label的一个成员，如果存在则直接返回，否则要计算，调用calculatePrefixView（内层模型定义的）
calculatePrefixView：在RC11中，获取的是porf之前的view （porf就是hb）
最后updata阶段，取并集，以及其他。首先遍历每个线程直到 porf 的prefix最大位置，- 然后如果某一个位置是 read，这个read在 preds中出现但是它的rf 没有在 porf prefix中出现，其它们在同一个线程，则从 preds 中去掉这个read的rf（read的rf一定在preds中出现了，因为preds取的是stamp小于read的）
- 如果一个位置是 rmw，其 porf prefix中有这个 rmw的前一个，则把这个rmw去掉
- 最后返回的是 preds，即读操作的

updata之后，就获得了一个 revisitview



```

```cpp
BackwardRevisit class:  public Revisit, public ReadRevisit
// 是一个 read 的 revisit，writerevist只是改变mo的位置

std::unique_ptr<VectorClock> view;      // 只有一个自己的成员，VectorClock只是一个基类，里面记录的是kind枚举类型

View 类型继承自 VectorClock ， 里面有一个成员
llvm::IndexedMap<int> EventView;
View的kind为 VC_View

```



# 如何cut
首先需要一个 read 和 write 的label，然后copygraph调用，然后，changerf

g.getPredsView(rLab->getPos()); 获取读操作的preds
calculatePrefixView(lab) 获取写操作的 porf 的prefix
或者getPrefixView，如果没有设置view则重新计算也可以
然后两者融合得到一个view

vectorclock: getMax(int thread) 获取当前线程最大值