import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from models.FM import FactorizationMachine

default_type = paddle.get_default_dtype()

class NRPA(nn.Layer):
    def __init__(self, user_num, item_num, vocab, args):
        super(NRPA, self).__init__()
        self.args = args
        self.u_embedding = nn.Embedding(user_num, args.id_embedding_dim)
        self.i_embedding = nn.Embedding(item_num, args.id_embedding_dim)
        self.user_review_net = ReviewEncoder(args=self.args, vocab=vocab)
        self.item_review_net = ReviewEncoder(args=self.args, vocab=vocab)

        self.user_net = UIEncoder(
            conv_kernel_num=args.conv_kernel_num, 
            id_embedding_dim=args.id_embedding_dim, 
            atten_vec_dim=args.atten_vec_dim
        )
        self.item_net = UIEncoder(
            conv_kernel_num=args.conv_kernel_num, 
            id_embedding_dim=args.id_embedding_dim, 
            atten_vec_dim=args.atten_vec_dim
        )
        self.fm = FactorizationMachine(args.conv_kernel_num * 2, args.fm_k)
    
    def forward(self, u_text, i_text, u_ids, i_ids):
        # u_text (u_batch, u_review_size, review_length)
        # i_text (i_batch, i_review_size, review_length)
        batch_size = u_text.shape[0]
        u_review_size, i_review_size = u_text.shape[1], i_text.shape[1]

        #User Module
        u_emb = self.u_embedding(u_ids) # user embedding
        d_matrix_user   = self.user_review_net(u_text, u_emb) #review encoder
        d_matrix_user   = d_matrix_user.reshape([batch_size, u_review_size, -1]).transpose([0,2,1])

        #Item Module
        i_emb = self.i_embedding(i_ids) #item embedding
        d_matrix_item   = self.item_review_net(i_text, i_emb) #review encoder
        d_matrix_item   = d_matrix_item.reshape([batch_size, i_review_size, -1]).transpose([0,2,1])

        pu = self.user_net(d_matrix_user, u_emb).squeeze(2)
        qi = self.item_net(d_matrix_item, i_emb).squeeze(2)
        
        x = paddle.concat((pu, qi), axis=1)
        rate = self.fm(x)
        return rate

class ReviewEncoder(nn.Layer):
    def __init__(self, args, vocab=None):
        super(ReviewEncoder, self).__init__()
        self.embedding_review = nn.Embedding(vocab.vocab_size, args.word_vec_dim)
        self.conv = nn.Conv1D( # input shape (batch_size, review_length, word_vec_dim)
            in_channels = args.word_vec_dim,
            out_channels = args.conv_kernel_num, 
            kernel_size = args.kernel_size,
            padding = (args.kernel_size -1) //2
        )# output shape (batch_size, conv_kernel_num, review_length)
        self.l1 = nn.Linear(args.id_embedding_dim, args.atten_vec_dim)
        self.A1 = paddle.create_parameter([args.atten_vec_dim, args.conv_kernel_num], default_type)

    def forward(self, review, ui_emb):
        # review (batch_size, review_size, review_length)
        # ui_emb (batch_size, ui_emb_dim)
        batch_size, review_size, review_length = review.shape
        new_batch       = batch_size * review_size
        review          = review.reshape([new_batch, -1])
        mul_ui_embs       = ui_emb.unsqueeze(1)
        ui_emb       = paddle.concat((mul_ui_embs,) * review_size, axis=1).reshape([new_batch, -1])
        review_vec      = self.embedding_review(review) #(batch_size, review_length, word_vec_dim)
        review_vec      = review_vec.transpose([0, 2, 1])
        # review_vec      = paddle.nn.functional.dropout(review_vec, 0.2, training=self.training)
        c               = F.relu(self.conv(review_vec)) #(batch_size, conv_kernel_num, review_length)
        c               = paddle.nn.functional.dropout(c, 0.05, training=self.training)
        qw              = F.relu(self.l1(ui_emb)) #(batch_size, atten_vec_dim)
        g               = paddle.matmul(qw, self.A1).unsqueeze(1) #(batch_size, 1, conv_kernel_num)
        g               = paddle.bmm(g, c) #(batch_size, 1, review_length)
        alph            = F.softmax(g, axis=2) #(batch_size, 1, review_length)
        d               = paddle.bmm(c, alph.transpose([0, 2, 1])) #(batch_size, conv_kernel_num, 1)
        # d               = torch.bmm(c, alph.transpose(0, 2, 1)) #(batch_size, conv_kernel_num, 1)
        return d

class UIEncoder(nn.Layer):
    def __init__(self, conv_kernel_num=80, atten_vec_dim=80, id_embedding_dim=32):
        # :param conv_kernel_num: 卷积核数量
        # :param id_matrix_len: id总个数
        # :param id_embedding_dim: id向量维度
        # :param atten_vec_dim: attention向量的维度
        super(UIEncoder, self).__init__()
        self.review_f = conv_kernel_num
        self.l1 = nn.Linear(id_embedding_dim, atten_vec_dim)
        self.A1 = paddle.create_parameter([atten_vec_dim, conv_kernel_num], default_type)

    def forward(self, review_emb, ui_emb):
        # now the batch_size = user_batch
        # word_Att => #(batch_size, conv_kernel_num, review_size)
        # id_vec          = self.embedding_id(ids) #(batch_size, id_embedding_dim)
        qr              = F.relu(self.l1(ui_emb)) #(batch_size, atten_vec_dim)
        e               = paddle.matmul(qr, self.A1).unsqueeze(1) #(batch_size, 1, conv_kernel_num)
        e               = paddle.bmm(e, review_emb) #(batch_size, 1, review_size)
        beta            = F.softmax(e, axis=2) #(batch_size, 1, review_size)
        p               = paddle.bmm(review_emb, beta.transpose([0, 2, 1])) #(batch_size, conv_kernel_num, 1)
        return p


