import tensorflow as tf

def main():
    inp = tf.constant(0)
     
    w = tf.constant(10)

    i1 = tf.Variable(0)
    h1 = tf.Variable(0)
    o1 = tf.Variable(0)

    i1 = inp
    h1 = i1 + w
    o1 = i1 + h1

    i2 = tf.Variable(0)
    h2 = tf.Variable(0)
    o2 = tf.Variable(0)

    i2 = o1
    h2 = i2 + w
    o2 = i2 + h2

    #init_op = tf.global_variables_initializer()
    
    with tf.Session() as sess:        
        #sess.run(init_op)
        sess.run(o2)
        print o2.eval()
        
if __name__ == '__main__':
    main()
